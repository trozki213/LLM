"""The seams. Every boundary that crosses a subsystem, a third party, or the filesystem.

Ports are `typing.Protocol`, not base classes: an adapter satisfies one by having the
right shape, so no infrastructure module ever has to import a framework to comply, and
no test double has to inherit anything. Dependencies are injected explicitly at the
composition root -- there is no registry, no singleton and no global here to find them.
"""

from __future__ import annotations

import datetime as dt
import typing

from fitkit.domain.body import BodyMeasurements, ScaleCalibration
from fitkit.domain.capture import CaptureBundle, PhotoRef
from fitkit.domain.contracts.explanation import Explanation
from fitkit.domain.contracts.fit_assessment import FitAssessment
from fitkit.domain.garment import GarmentSpec
from fitkit.domain.regions import BodyRegion


class ScaleCalibrationSource(typing.Protocol):
    """Where metric scale comes from. C3 lives here: declared height now, depth later."""

    source_id: str

    def calibrate(self, bundle: CaptureBundle) -> ScaleCalibration: ...


class MeasurementBackend(typing.Protocol):
    """C5: assume this will be replaced. Every implementation passes one contract suite."""

    backend_id: str
    supported_regions: frozenset[BodyRegion]

    def estimate(
        self, bundle: CaptureBundle, calibration: ScaleCalibration
    ) -> BodyMeasurements: ...


class GarmentRepository(typing.Protocol):
    def get(self, garment_id: str, version: int | None = None) -> GarmentSpec: ...

    def latest_version(self, garment_id: str) -> int: ...


class ExplanationRenderer(typing.Protocol):
    """Takes the assessment and nothing else. Locale and tone travel in render_hints."""

    renderer_id: str

    def render(self, assessment: FitAssessment) -> Explanation: ...


class LlmClient(typing.Protocol):
    """The only place an LLM is reachable from. Returns text; it decides nothing."""

    def complete(self, prompt: str, *, max_tokens: int) -> str: ...


class PhotoStore(typing.Protocol):
    """Images are the sensitive asset; deletion is part of the interface, not an afterthought."""

    def put(self, capture_id: str, view: str, data: bytes) -> PhotoRef: ...

    def get(self, ref: PhotoRef) -> bytes: ...

    def delete(self, ref: PhotoRef) -> None: ...


class AssessmentStore(typing.Protocol):
    """Persisted assessments are the audit record and the evaluation substrate."""

    def save(self, assessment: FitAssessment) -> None: ...

    def load(self, assessment_id: str) -> FitAssessment: ...


class Clock(typing.Protocol):
    """Injected so the engine and its tests never read the wall clock."""

    def now(self) -> dt.datetime: ...


class MetricsPort(typing.Protocol):
    def increment(self, name: str, tags: dict[str, str] | None = None) -> None: ...

    def observe(self, name: str, value: float, tags: dict[str, str] | None = None) -> None: ...
