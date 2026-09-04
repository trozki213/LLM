"""The public operation, and the only place that knows the pipeline's shape.

Facade: one entry point over five subsystems. It is a facade because the system really
does have five subsystems, not because a pattern was wanted -- without it every caller
reassembles the ordering, the degradation ladder and the persistence rules.

Every dependency arrives through the constructor. No singletons, no service locator, no
module-level state, no framework magic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Mapping

from fitkit.domain.capture import CaptureBundle
from fitkit.domain.contracts.explanation import Explanation
from fitkit.domain.contracts.fit_assessment import AbstainCode, Coverage, FitAssessment, Verdict
from fitkit.domain.errors import DegradationCode, StorageError
from fitkit.domain.policy import FitPolicy, FitPreference, Tone
from fitkit.domain.ports import (
    AssessmentStore,
    Clock,
    ExplanationRenderer,
    GarmentRepository,
    MeasurementBackend,
    MetricsPort,
    ScaleCalibrationSource,
)
from fitkit.fit_engine.engine import DeterministicFitEngine


@dataclass(frozen=True, slots=True)
class AdviceRequest:
    capture: CaptureBundle
    garment_id: str
    merchant_id: str
    preference: FitPreference = FitPreference.AS_DESIGNED
    locale: str = "en"
    tone: Tone = Tone.NEUTRAL


@dataclass(frozen=True, slots=True)
class AdviceResult:
    assessment: FitAssessment
    explanation: Explanation
    degradations: tuple[DegradationCode, ...]

    @property
    def assessment_id(self) -> str:
        """The join key that eventually reaches the order record (design, Phase 7)."""
        return self.assessment.assessment_id


class NullMetrics:
    def increment(self, name: str, tags: dict[str, str] | None = None) -> None: ...

    def observe(self, name: str, value: float, tags: dict[str, str] | None = None) -> None: ...


class SizeAdvisor:
    def __init__(
        self,
        *,
        calibration: ScaleCalibrationSource,
        backend: MeasurementBackend,
        garments: GarmentRepository,
        engine: DeterministicFitEngine,
        renderer: ExplanationRenderer,
        store: AssessmentStore,
        clock: Clock,
        policies: Mapping[str, FitPolicy],
        default_policy: FitPolicy,
        metrics: MetricsPort | None = None,
        id_factory: Callable[[AdviceRequest, str, str], str] | None = None,
    ) -> None:
        self._calibration = calibration
        self._backend = backend
        self._garments = garments
        self._engine = engine
        self._renderer = renderer
        self._store = store
        self._clock = clock
        self._policies = dict(policies)
        self._default_policy = default_policy
        self._metrics = metrics or NullMetrics()
        self._id_factory = id_factory or deterministic_assessment_id

    def advise(self, request: AdviceRequest) -> AdviceResult:
        garment = self._garments.get(request.garment_id)
        policy = self._policies.get(request.merchant_id, self._default_policy)
        assessment_id = self._id_factory(request, garment.version_key, policy.version_key)

        cached = self._load(assessment_id)
        if cached is not None:
            self._metrics.increment("advise.idempotent_hit")
            return self._finish(cached)

        calibration = self._calibration.calibrate(request.capture)
        body = self._backend.estimate(request.capture, calibration)
        self._metrics.observe(
            "measurement.scale_sigma_rel", body.scale_sigma_rel, {"backend": self._backend.backend_id}
        )

        assessment = self._engine.assess(
            body=body,
            garment=garment,
            preference=request.preference,
            policy=policy,
            assessment_id=assessment_id,
            computed_at=self._clock.now(),
            locale=request.locale,
            tone=request.tone,
        )
        self._store.save(assessment)
        return self._finish(assessment)

    # -- internals ---------------------------------------------------------------

    def _finish(self, assessment: FitAssessment) -> AdviceResult:
        # The explanation is always re-rendered, including on the idempotent path: it is
        # cheap, it is derived purely from the assessment, and it is not persisted. The
        # assessment is the audit record; the prose is not.
        explanation = self._renderer.render(assessment)
        degradations = _degradations(assessment, explanation)
        self._metrics.increment(
            "advise.verdict", {"verdict": assessment.recommendation.verdict.value}
        )
        if explanation.degraded:
            # A rising count here means a renderer is trying to say things the contract
            # does not support. That is a safety signal, not a quality one.
            self._metrics.increment("explanation.degraded", {"renderer": explanation.renderer_id})
        return AdviceResult(assessment, explanation, degradations)

    def _load(self, assessment_id: str) -> FitAssessment | None:
        try:
            return self._store.load(assessment_id)
        except StorageError:
            return None


def _degradations(
    assessment: FitAssessment, explanation: Explanation
) -> tuple[DegradationCode, ...]:
    codes: list[DegradationCode] = []
    rec = assessment.recommendation
    if rec.verdict is Verdict.ABSTAIN and rec.abstain.code in (
        AbstainCode.UNCERTAINTY_EXCEEDS_SIZE_STEP,
        AbstainCode.INSUFFICIENT_BODY_DATA,
    ):
        codes.append(DegradationCode.MEASUREMENT_UNCERTAIN)
    if any(size.coverage is Coverage.PARTIAL for size in assessment.sizes):
        codes.append(DegradationCode.COVERAGE_PARTIAL)
    if explanation.degraded:
        codes.append(DegradationCode.EXPLANATION_TEMPLATED)
    return tuple(codes)


def deterministic_assessment_id(
    request: AdviceRequest, garment_version_key: str, policy_version_key: str
) -> str:
    """A content-addressed id.

    Deterministic rather than random so the same inputs are idempotent, and so Phase 7's
    replay can address a historical assessment without a lookup table.
    """
    material = "|".join(
        (
            request.capture.capture_id,
            garment_version_key,
            policy_version_key,
            request.preference.value,
            request.locale,
        )
    )
    return hashlib.sha256(material.encode()).hexdigest()[:26].upper()
