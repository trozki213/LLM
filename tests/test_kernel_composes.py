"""The whole pipeline shape, expressed with kernel types and fakes only.

There is no fit engine and no renderer yet. The point of this test is to prove the
vocabulary is *sufficient* -- that capture, calibration, measurement, assessment and
explanation compose without a missing type -- while it is still cheap to fix. A gap
discovered here costs an afternoon; the same gap discovered in Phase 6 costs a contract
version.
"""

from __future__ import annotations

import datetime as dt

from fitkit.domain.capture import CaptureBundle, DeviceMetadata, GateVerdict, PhotoRef
from fitkit.domain.contracts.fit_assessment import (
    Coverage,
    FabricSummary,
    FitAssessment,
    FitClassification,
    GarmentRef,
    InputsDigest,
    Recommendation,
    RegionDelta,
    RenderHints,
    SizeAssessment,
    SizeChoice,
    Verdict,
)
from fitkit.domain.fabric import FabricSpec, RecoveryClass, StretchClass
from fitkit.domain.garment import GarmentSizeSpec, GarmentSpec
from fitkit.domain.policy import EaseWindow, FitPolicy, FitPreference, Tone
from fitkit.domain.regions import BodyRegion, FitIntent, GarmentCategory, GarmentRegion
from fitkit.domain.units import Measure, MeasureSource

from tests import fakes


def _bundle() -> CaptureBundle:
    return CaptureBundle(
        capture_id="cap_01J",
        frontal=PhotoRef(uri="memory://cap_01J/frontal", sha256="a" * 64),
        lateral=PhotoRef(uri="memory://cap_01J/lateral", sha256="b" * 64),
        declared_height=Measure(175.0, 1.5, MeasureSource.USER_DECLARED),
        declared_weight=None,
        device=DeviceMetadata(platform="ios", model="iPhone15,2", app_version="1.0.0"),
        gate_report=(GateVerdict("framing", True, 0.94, None),),
    )


def _garment() -> GarmentSpec:
    def size(label: str, waist: float, hip: float) -> GarmentSizeSpec:
        return GarmentSizeSpec(
            size_label=label,
            measurements={
                GarmentRegion.WAIST_FLAT: Measure(waist, 0.5, MeasureSource.SPEC_SHEET),
                GarmentRegion.HIP_FLAT: Measure(hip, 0.5, MeasureSource.SPEC_SHEET),
            },
        )

    return GarmentSpec(
        garment_id="brand:sku-1",
        version=7,
        category=GarmentCategory.TROUSERS,
        size_system="EU",
        fit_intent=FitIntent.REGULAR,
        fabric=FabricSpec(stretch_class=StretchClass.NONE, recovery=RecoveryClass.GOOD),
        sizes=(size("46", 38.0, 46.0), size("48", 40.0, 48.0)),
    )


POLICY = FitPolicy(
    policy_id="policy/merchant-a",
    version=3,
    tau_single=0.65,
    tau_pair=0.85,
    max_critical_sigma_cm=2.5,
    region_weights={BodyRegion.WAIST: 1.0, BodyRegion.HIP: 0.8},
    critical_regions=frozenset({BodyRegion.WAIST, BodyRegion.HIP}),
    tightness_penalty_ratio=1.8,
)


def test_the_kernel_expresses_the_whole_pipeline():
    bundle = _bundle()

    # Acquisition -> calibration (the C3 seam)
    calibration = fakes.DeclaredHeightCalibrationStub().calibrate(bundle)
    assert calibration.source_id == "declared-height"

    # Calibration -> measurement (the C5 seam)
    body = fakes.FixedMeasurementBackend(waist_cm=82.0, hip_cm=96.0).estimate(bundle, calibration)

    # The correlated-error model is doing work: the total exceeds the residual.
    assert body[BodyRegion.WAIST].sigma_cm > body.residual_sigma_cm(BodyRegion.WAIST)

    # Garment side: a flat spec measure becomes a comparable girth, sigma included.
    garment = _garment()
    flat = garment.size("48").measurements[GarmentRegion.WAIST_FLAT]
    girth = flat.scaled(2.0)
    assert girth.value_cm == 80.0
    assert girth.sigma_cm == 1.0  # 0.5 flat tolerance, doubled

    # The comparison the fit engine will make in Phase 4, with propagated uncertainty.
    delta = girth - body[BodyRegion.WAIST]
    assert delta.value_cm < 0  # 80 cm garment on an 82 cm waist: tight
    assert delta.sigma_cm > girth.sigma_cm
    assert delta.source is MeasureSource.DERIVED

    # A hand-built assessment stands in for the engine's output.
    assessment = FitAssessment(
        assessment_id="01JBQ7H3M4N5P6Q7R8S9T0V1W2",
        garment=GarmentRef(
            garment_id=garment.garment_id,
            category=garment.category,
            size_system=garment.size_system,
            fit_intent=garment.fit_intent,
        ),
        fabric=FabricSummary(StretchClass.NONE, RecoveryClass.GOOD, 0.0),
        recommendation=Recommendation(
            verdict=Verdict.TWO_SIZES,
            primary=SizeChoice("48", 0.54),
            alternate=SizeChoice("46", 0.31),
            abstain=None,
        ),
        sizes=(
            SizeAssessment(
                size_label="48",
                confidence=0.54,
                regions=(
                    RegionDelta(
                        region=BodyRegion.WAIST,
                        critical=True,
                        delta_cm=round(delta.value_cm, 2),
                        delta_sigma_cm=round(delta.sigma_cm, 2),
                        stretch_absorbed_cm=0.0,
                        required_ease=EaseWindow(1.0, 2.0, 5.0),
                        classification=FitClassification.TIGHT,
                        uncertain=True,
                    ),
                ),
                coverage=Coverage.COMPLETE,
                missing_regions=(),
            ),
            SizeAssessment("46", 0.31, (), Coverage.COMPLETE, ()),
        ),
        inputs_digest=InputsDigest(
            measurement_backend=f"{body.provenance.backend_id}@{body.provenance.backend_version}",
            measurement_provenance_id=body.provenance.capture_id,
            garment_spec_version=garment.version_key,
            engine_version="fit-engine/0.0.0",
            policy_version=POLICY.version_key,
            residual_table_version=body.provenance.residual_table_version,
            computed_at=fakes.FixedClock(dt.datetime(2026, 9, 3, tzinfo=dt.UTC)).now(),
        ),
        render_hints=RenderHints(locale="it-IT", tone=Tone.NEUTRAL),
    )

    # Every input that produced it is recoverable from the document. Phase 7 depends on this.
    assert assessment.inputs_digest.garment_spec_version == "brand:sku-1@7"
    assert assessment.inputs_digest.policy_version == "policy/merchant-a/3"

    # Assessment -> explanation (the C1/C2 seam). The renderer sees the document, nothing else.
    explanation = fakes.StubRenderer().render(assessment)
    assert explanation.degraded is False

    # And the guard Phase 5 will run has everything it needs, from the document alone.
    allowed = assessment.numeric_allowlist()
    assert abs(round(delta.value_cm, 2)) in allowed
    assert 48.0 in allowed
    assert 999.0 not in allowed


def test_a_preference_is_carried_without_the_kernel_interpreting_it():
    """FitPreference is vocabulary here; turning it into ease is Phase 4's job."""
    assert FitPreference.LOOSER in set(FitPreference)
    assert POLICY.region_weights[BodyRegion.WAIST] == 1.0
