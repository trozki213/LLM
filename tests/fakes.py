"""Test doubles the ports make possible. Phases 2 and 5 build their suites on these.

Every fake here is a plain object. None of them inherits from anything, which is the
practical payoff of using Protocols rather than base classes at the seams.
"""

from __future__ import annotations

import datetime as dt

from fitkit.domain.body import BodyMeasurements, MeasurementProvenance, ScaleCalibration
from fitkit.domain.capture import CaptureBundle, PhotoRef
from fitkit.domain.contracts.explanation import Explanation
from fitkit.domain.contracts.fit_assessment import FitAssessment
from fitkit.domain.errors import BackendUnavailable, GarmentNotFound, StorageError
from fitkit.domain.garment import GarmentSpec
from fitkit.domain.regions import BodyRegion
from fitkit.domain.units import Measure, MeasureSource


class FixedClock:
    def __init__(self, moment: dt.datetime) -> None:
        self._moment = moment

    def now(self) -> dt.datetime:
        return self._moment


class DeclaredHeightCalibrationStub:
    source_id = "declared-height"

    def __init__(self, sigma_rel: float = 0.009) -> None:
        self._sigma_rel = sigma_rel

    def calibrate(self, bundle: CaptureBundle) -> ScaleCalibration:
        return ScaleCalibration(
            source_id=self.source_id,
            sigma_rel=self._sigma_rel,
            reference=bundle.declared_height,
        )


class FixedMeasurementBackend:
    """Returns a canned body. The workhorse double for the engine and orchestration."""

    backend_id = "fixed"
    supported_regions = frozenset({BodyRegion.WAIST, BodyRegion.HIP})

    def __init__(self, waist_cm: float = 80.0, hip_cm: float = 95.0, residual_cm: float = 1.2) -> None:
        self._waist, self._hip, self._residual = waist_cm, hip_cm, residual_cm

    def estimate(self, bundle: CaptureBundle, calibration: ScaleCalibration) -> BodyMeasurements:
        return BodyMeasurements(
            residuals={
                BodyRegion.WAIST: Measure(self._waist, self._residual, MeasureSource.ESTIMATED),
                BodyRegion.HIP: Measure(self._hip, self._residual, MeasureSource.ESTIMATED),
            },
            scale_sigma_rel=calibration.sigma_rel,
            provenance=MeasurementProvenance(
                backend_id=self.backend_id,
                backend_version="0",
                residual_table_version="residuals/fake",
                capture_id=bundle.capture_id,
                calibration_source_id=calibration.source_id,
                computed_at=dt.datetime(2026, 9, 3, tzinfo=dt.UTC),
            ),
        )


class PerturbingMeasurementBackend(FixedMeasurementBackend):
    """Adds a known error, so abstention can be exercised end to end (design, Phase 2)."""

    backend_id = "perturbing"

    def __init__(self, *, residual_cm: float, **kwargs) -> None:
        super().__init__(residual_cm=residual_cm, **kwargs)


class FailingMeasurementBackend:
    backend_id = "failing"
    supported_regions: frozenset[BodyRegion] = frozenset()

    def estimate(self, bundle: CaptureBundle, calibration: ScaleCalibration) -> BodyMeasurements:
        raise BackendUnavailable("the fake backend is down")


class InMemoryGarmentRepository:
    def __init__(self, *specs: GarmentSpec) -> None:
        self._by_key = {(s.garment_id, s.version): s for s in specs}

    def get(self, garment_id: str, version: int | None = None) -> GarmentSpec:
        if version is None:
            version = self.latest_version(garment_id)
        try:
            return self._by_key[(garment_id, version)]
        except KeyError:
            raise GarmentNotFound(f"{garment_id}@{version}") from None

    def latest_version(self, garment_id: str) -> int:
        versions = [v for (gid, v) in self._by_key if gid == garment_id]
        if not versions:
            raise GarmentNotFound(garment_id)
        return max(versions)


class StubRenderer:
    renderer_id = "stub/1"

    def __init__(self, text: str = "A stub explanation.", degraded: bool = False) -> None:
        self._text, self._degraded = text, degraded

    def render(self, assessment: FitAssessment) -> Explanation:
        return Explanation(text=self._text, renderer_id=self.renderer_id, degraded=self._degraded)


class ScriptedLlmClient:
    """Returns whatever it was told to. Adversarial scripts are how the guard is proven."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.calls: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        self.calls.append(prompt)
        return self._replies.pop(0) if self._replies else ""


class InMemoryPhotoStore:
    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}

    def put(self, capture_id: str, view: str, data: bytes) -> PhotoRef:
        import hashlib

        digest = hashlib.sha256(data).hexdigest()
        uri = f"memory://{capture_id}/{view}"
        self._blobs[uri] = data
        return PhotoRef(uri=uri, sha256=digest)

    def get(self, ref: PhotoRef) -> bytes:
        try:
            return self._blobs[ref.uri]
        except KeyError:
            raise StorageError(ref.uri) from None

    def delete(self, ref: PhotoRef) -> None:
        self._blobs.pop(ref.uri, None)


class InMemoryAssessmentStore:
    def __init__(self) -> None:
        self.saved: dict[str, FitAssessment] = {}

    def save(self, assessment: FitAssessment) -> None:
        self.saved[assessment.assessment_id] = assessment

    def load(self, assessment_id: str) -> FitAssessment:
        try:
            return self.saved[assessment_id]
        except KeyError:
            raise StorageError(assessment_id) from None


class RecordingMetrics:
    def __init__(self) -> None:
        self.counters: list[tuple[str, dict[str, str] | None]] = []
        self.observations: list[tuple[str, float, dict[str, str] | None]] = []

    def increment(self, name: str, tags: dict[str, str] | None = None) -> None:
        self.counters.append((name, tags))

    def observe(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        self.observations.append((name, value, tags))
