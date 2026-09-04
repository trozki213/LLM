"""Orchestration: the full pipeline with every port faked, and the degradation ladder."""
import datetime as dt

import pytest

from fitkit.acquisition import CaptureAssembler, RawCapture
from fitkit.catalog import CsvSpecImporter, GarmentSpecBuilder, InMemoryGarmentRepository
from fitkit.domain.capture import DeviceMetadata, FrameSignals, ViewKind
from fitkit.domain.errors import BackendUnavailable, DegradationCode, GarmentNotFound
from fitkit.domain.fabric import FabricSpec, RecoveryClass, StretchClass
from fitkit.domain.policy import FitPolicy, FitPreference
from fitkit.domain.regions import BodyRegion, FitIntent, GarmentCategory
from fitkit.evaluation import GroundTruthSample, build_residual_table
from fitkit.measurement import ResidualEntry, ResidualTable
from fitkit.orchestration import AdviceRequest, SizeAdvisor, build_advisor, deterministic_assessment_id

from tests.fakes import (
    FailingMeasurementBackend,
    FixedClock,
    FixedMeasurementBackend,
    InMemoryAssessmentStore,
    InMemoryPhotoStore,
    RecordingMetrics,
    ScriptedLlmClient,
)

NOW = dt.datetime(2026, 9, 3, 11, 4, 22, tzinfo=dt.UTC)
CSV = b"""size_label,region,value_cm,tolerance_cm
46,waist_flat,39.0,0.6
46,hip_flat,47.0,0.6
48,waist_flat,41.0,0.6
48,hip_flat,49.0,0.6
50,waist_flat,43.0,0.6
50,hip_flat,51.0,0.6
"""
RESIDUALS = ResidualTable(
    version="residuals/2026-09",
    entries=(
        ResidualEntry("fixed", BodyRegion.WAIST, 999.0, 1.1),
        ResidualEntry("fixed", BodyRegion.HIP, 999.0, 1.0),
    ),
)
POLICY = FitPolicy(
    policy_id="policy/merchant-a", version=3, tau_single=0.65, tau_pair=0.85,
    max_critical_sigma_cm=2.5,
    region_weights={BodyRegion.WAIST: 1.0, BodyRegion.HIP: 0.8},
    critical_regions=frozenset({BodyRegion.WAIST, BodyRegion.HIP}),
    tightness_penalty_ratio=1.8,
)


class StubAnalyzer:
    analyzer_id = "stub/1"

    def analyze(self, image, view):
        return FrameSignals(
            view=view, head_visible=True, feet_visible=True, subject_frame_fraction=0.78,
            sharpness=0.9, exposure=0.7, background_separability=0.8, arm_separation=0.8,
            torso_verticality=0.95, device_pitch_deg=2.0, clothing_tightness=0.8,
        )


@pytest.fixture
def garments():
    spec = (
        GarmentSpecBuilder()
        .with_identity("brand:sku-1", version=7, category=GarmentCategory.TROUSERS,
                       size_system="EU", fit_intent=FitIntent.REGULAR)
        .with_fabric(FabricSpec(StretchClass.LOW, RecoveryClass.GOOD))
        .with_grading_tolerance(0.6)
        .with_rows(CsvSpecImporter().parse(CSV).rows)
        .build()
    )
    return InMemoryGarmentRepository(spec)


@pytest.fixture
def capture():
    assembler = CaptureAssembler(StubAnalyzer(), InMemoryPhotoStore())
    return assembler.assemble(
        "cap_01J",
        RawCapture(b"frontal", b"lateral", 175.0, None, DeviceMetadata("ios", "x", "1.0.0")),
    )


def advisor(garments, *, backend=None, llm=None, store=None, metrics=None):
    return build_advisor(
        garments=garments,
        store=store or InMemoryAssessmentStore(),
        residuals=RESIDUALS,
        default_policy=POLICY,
        backend=backend or FixedMeasurementBackend(waist_cm=82.0, hip_cm=97.0),
        llm_client=llm,
        clock=FixedClock(NOW),
        metrics=metrics,
    )


def request(**kw):
    fields = dict(garment_id="brand:sku-1", merchant_id="merchant-a",
                  preference=FitPreference.AS_DESIGNED, locale="en")
    fields.update(kw)
    return AdviceRequest(**fields)


class TestHappyPath:
    def test_produces_an_assessment_and_an_explanation(self, garments, capture):
        result = advisor(garments).advise(request(capture=capture))
        assert result.assessment.sizes
        assert result.explanation.text
        assert result.degradations == ()

    def test_runs_with_no_network_and_no_database(self, garments, capture):
        """Every port faked: the full pipeline is a unit test."""
        result = advisor(garments).advise(request(capture=capture))
        assert result.assessment.inputs_digest.residual_table_version == "residuals/2026-09"

    def test_persists_exactly_one_assessment(self, garments, capture):
        store = InMemoryAssessmentStore()
        result = advisor(garments, store=store).advise(request(capture=capture))
        assert list(store.saved) == [result.assessment_id]

    def test_returns_the_join_key_for_the_order_record(self, garments, capture):
        result = advisor(garments).advise(request(capture=capture))
        assert result.assessment_id == result.assessment.assessment_id

    def test_records_the_verdict_as_a_metric(self, garments, capture):
        metrics = RecordingMetrics()
        advisor(garments, metrics=metrics).advise(request(capture=capture))
        assert any(name == "advise.verdict" for name, _ in metrics.counters)


class TestIdempotency:
    def test_the_same_request_returns_the_stored_assessment(self, garments, capture):
        store = InMemoryAssessmentStore()
        service = advisor(garments, store=store)
        first = service.advise(request(capture=capture))
        second = service.advise(request(capture=capture))
        assert first.assessment == second.assessment
        assert len(store.saved) == 1

    def test_a_different_preference_is_a_different_assessment(self, garments, capture):
        service = advisor(garments)
        a = service.advise(request(capture=capture))
        b = service.advise(request(capture=capture, preference=FitPreference.LOOSER))
        assert a.assessment_id != b.assessment_id

    def test_the_id_is_derived_from_the_inputs_not_from_a_clock(self, garments, capture):
        req = request(capture=capture)
        assert deterministic_assessment_id(req, "brand:sku-1@7", "policy/merchant-a/3") == (
            deterministic_assessment_id(req, "brand:sku-1@7", "policy/merchant-a/3")
        )

    def test_a_new_garment_version_produces_a_new_assessment(self, garments, capture):
        req = request(capture=capture)
        assert deterministic_assessment_id(req, "brand:sku-1@7", "p/1") != (
            deterministic_assessment_id(req, "brand:sku-1@8", "p/1")
        )


class TestDegradationLadder:
    def test_an_unavailable_llm_degrades_the_prose_not_the_numbers(self, garments, capture):
        class DeadLlm:
            def complete(self, prompt, *, max_tokens):
                raise BackendUnavailable("down")

        result = advisor(garments, llm=DeadLlm()).advise(request(capture=capture))
        assert DegradationCode.EXPLANATION_TEMPLATED in result.degradations
        assert result.assessment.recommendation.verdict.value in ("SINGLE", "TWO_SIZES", "ABSTAIN")
        assert result.explanation.text

    def test_a_lying_llm_is_replaced_by_the_template(self, garments, capture):
        llm = ScriptedLlmClient("You have 37.5 cm of room.")
        result = advisor(garments, llm=llm).advise(request(capture=capture))
        assert result.explanation.degraded is True
        assert "37.5" not in result.explanation.text

    def test_a_clean_llm_answer_is_used(self, garments, capture):
        good = "Order the 48. The waist sits about 2 cm looser than intended."
        result = advisor(garments, llm=ScriptedLlmClient(good)).advise(request(capture=capture))
        assert result.explanation.degraded is False

    def test_an_uncertain_measurement_reaches_the_user_as_a_degradation(self, garments, capture):
        noisy = ResidualTable(
            version="residuals/noisy",
            entries=(
                ResidualEntry("fixed", BodyRegion.WAIST, 999.0, 4.0),
                ResidualEntry("fixed", BodyRegion.HIP, 999.0, 4.0),
            ),
        )
        service = build_advisor(
            garments=garments, store=InMemoryAssessmentStore(), residuals=noisy,
            default_policy=POLICY, backend=FixedMeasurementBackend(waist_cm=82.0, hip_cm=97.0),
            clock=FixedClock(NOW),
        )
        result = service.advise(request(capture=capture))
        assert result.assessment.recommendation.verdict.value == "ABSTAIN"
        assert DegradationCode.MEASUREMENT_UNCERTAIN in result.degradations

    def test_a_dead_backend_fails_rather_than_fabricating_measurements(self, garments, capture):
        with pytest.raises(BackendUnavailable):
            advisor(garments, backend=FailingMeasurementBackend()).advise(request(capture=capture))

    def test_an_unknown_garment_is_a_typed_input_error(self, garments, capture):
        with pytest.raises(GarmentNotFound):
            advisor(garments).advise(request(capture=capture, garment_id="brand:nope"))


class TestC2WithoutAnLlm:
    def test_the_default_wiring_uses_the_template(self, garments, capture):
        result = advisor(garments).advise(request(capture=capture))
        assert result.explanation.renderer_id == "template/1"
        assert result.explanation.degraded is False


class TestCompositionRoot:
    def test_wires_a_vendor_backend_from_a_transport(self, garments, capture):
        class StubTransport:
            def post_json(self, url, payload, *, timeout_s):
                return {
                    "measurements": {"waist": 82.0, "hip": 97.0},
                    "computed_at": "2026-09-03T11:04:22+00:00",
                }

        service = build_advisor(
            garments=garments, store=InMemoryAssessmentStore(),
            residuals=ResidualTable(
                version="residuals/vendor",
                entries=(
                    ResidualEntry("vendor", BodyRegion.WAIST, 999.0, 1.2),
                    ResidualEntry("vendor", BodyRegion.HIP, 999.0, 1.1),
                ),
            ),
            default_policy=POLICY, transport=StubTransport(),
            vendor_url="https://vendor.example/scan", vendor_version="2026.07",
            clock=FixedClock(NOW),
        )
        result = service.advise(request(capture=capture))
        assert result.assessment.inputs_digest.measurement_backend == "vendor@2026.07"

    def test_it_refuses_to_wire_a_system_with_no_measurement_source(self, garments):
        with pytest.raises(ValueError, match="backend or an HTTP transport"):
            build_advisor(
                garments=garments, store=InMemoryAssessmentStore(),
                residuals=RESIDUALS, default_policy=POLICY,
            )

    def test_a_merchant_policy_overrides_the_default(self, garments, capture):
        strict = FitPolicy(
            policy_id="policy/merchant-b", version=1, tau_single=0.99, tau_pair=0.999,
            max_critical_sigma_cm=0.5,
            region_weights={BodyRegion.WAIST: 1.0, BodyRegion.HIP: 0.8},
            critical_regions=frozenset({BodyRegion.WAIST}),
            tightness_penalty_ratio=1.8,
        )
        service = build_advisor(
            garments=garments, store=InMemoryAssessmentStore(), residuals=RESIDUALS,
            default_policy=POLICY, policies={"merchant-b": strict},
            backend=FixedMeasurementBackend(waist_cm=82.0, hip_cm=97.0), clock=FixedClock(NOW),
        )
        result = service.advise(request(capture=capture, merchant_id="merchant-b"))
        assert result.assessment.inputs_digest.policy_version == "policy/merchant-b/1"
        assert result.assessment.recommendation.verdict.value == "ABSTAIN"

    def test_the_system_clock_returns_timezone_aware_time(self):
        from fitkit.orchestration import SystemClock

        assert SystemClock().now().tzinfo is not None
