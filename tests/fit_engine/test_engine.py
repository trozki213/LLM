"""The fit engine: pure, deterministic, and the only place a size is chosen."""
import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fitkit.domain.contracts.fit_assessment import (
    AbstainCode,
    Coverage,
    FitAssessment,
    FitClassification,
    Verdict,
)
from fitkit.domain.policy import FitPreference
from fitkit.domain.regions import BodyRegion, FitIntent

from tests.fit_engine.conftest import BAGGY, JERSEY, RIGID, assess, body, policy, trousers


class TestOutputIsAValidContract:
    def test_produces_a_document_that_round_trips(self, engine):
        a = assess(engine)
        assert FitAssessment.from_dict(a.to_dict()) == a

    def test_assesses_every_size_in_the_run(self, engine):
        assert {s.size_label for s in assess(engine).sizes} == {"46", "48", "50"}

    def test_confidences_form_a_distribution(self, engine):
        total = sum(s.confidence for s in assess(engine).sizes)
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_records_exactly_what_produced_it(self, engine):
        digest = assess(engine).inputs_digest
        assert digest.garment_spec_version == "brand:sku-1@7"
        assert digest.policy_version == "policy/merchant-a/3"
        assert digest.measurement_backend == "fixed@1"
        assert digest.residual_table_version == "residuals/1"
        assert "fit-engine/1.0.0" in digest.engine_version


class TestDeterminism:
    def test_identical_inputs_give_a_byte_identical_document(self, engine):
        first = json.dumps(assess(engine).to_dict(), sort_keys=True)
        for _ in range(200):
            assert json.dumps(assess(engine).to_dict(), sort_keys=True) == first

    def test_a_fresh_engine_instance_gives_the_same_answer(self):
        from fitkit.fit_engine import DeterministicFitEngine

        a = json.dumps(assess(DeterministicFitEngine()).to_dict(), sort_keys=True)
        b = json.dumps(assess(DeterministicFitEngine()).to_dict(), sort_keys=True)
        assert a == b

    def test_region_order_is_stable(self, engine):
        for size in assess(engine).sizes:
            names = [d.region.name for d in size.regions]
            assert names == sorted(names)


class TestFabricChangesTheOutcome:
    """C4: the same geometry on different cloth is a different answer."""

    def _waist_delta(self, engine, fabric, label="46"):
        a = assess(engine, garment=trousers(fabric=fabric), body=body(waist=82.0))
        size = next(s for s in a.sizes if s.size_label == label)
        return next(d for d in size.regions if d.region is BodyRegion.WAIST)

    def test_rigid_denim_and_jersey_classify_the_same_shortfall_differently(self, engine):
        rigid = self._waist_delta(engine, RIGID)
        jersey = self._waist_delta(engine, JERSEY)
        assert rigid.delta_cm < jersey.delta_cm
        assert rigid.classification is not jersey.classification

    def test_stretch_absorbs_only_a_shortfall(self, engine):
        """A garment that already fits is not made looser by being stretchy."""
        loose = assess(engine, garment=trousers(fabric=JERSEY), body=body(waist=70.0))
        size = next(s for s in loose.sizes if s.size_label == "50")
        waist = next(d for d in size.regions if d.region is BodyRegion.WAIST)
        assert waist.stretch_absorbed_cm == 0.0

    def test_poor_recovery_buys_much_less_than_good_recovery(self, engine):
        good = self._waist_delta(engine, JERSEY)
        poor = self._waist_delta(engine, BAGGY)
        assert poor.delta_cm < good.delta_cm

    def test_rigid_fabric_absorbs_nothing(self, engine):
        assert self._waist_delta(engine, RIGID).stretch_absorbed_cm == 0.0

    def test_the_document_reports_the_usable_extension_it_used(self, engine):
        assert assess(engine, garment=trousers(fabric=RIGID)).fabric.usable_extension_pct == 0.0
        assert assess(engine, garment=trousers(fabric=JERSEY)).fabric.usable_extension_pct > 0.0


class TestUncertaintyDrivesTheVerdict:
    """C6, executable."""

    def test_a_confident_body_gets_a_single_size(self, engine):
        a = assess(engine, body=body(waist=80.0, hip=95.0, sigma=0.4, scale=0.003))
        assert a.recommendation.verdict is Verdict.SINGLE

    def test_three_centimetre_sigma_against_a_four_centimetre_step_never_commits(self, engine):
        """The single most important test in the system."""
        a = assess(engine, body=body(sigma=3.0))
        assert a.recommendation.verdict in (Verdict.TWO_SIZES, Verdict.ABSTAIN)

    def test_uncertainty_beyond_the_ceiling_abstains_with_a_reason(self, engine):
        a = assess(engine, body=body(sigma=3.0), policy=policy(max_critical_sigma_cm=2.0))
        assert a.recommendation.verdict is Verdict.ABSTAIN
        assert a.recommendation.abstain.code is AbstainCode.UNCERTAINTY_EXCEEDS_SIZE_STEP
        assert a.recommendation.abstain.detail_codes

    def test_abstaining_still_reports_every_size(self, engine):
        a = assess(engine, body=body(sigma=3.0), policy=policy(max_critical_sigma_cm=2.0))
        assert len(a.sizes) == 3
        assert a.recommendation.primary is None

    def test_more_uncertainty_never_increases_confidence(self, engine):
        sharp = max(s.confidence for s in assess(engine, body=body(sigma=0.4)).sizes)
        blurry = max(s.confidence for s in assess(engine, body=body(sigma=2.4)).sizes)
        assert blurry <= sharp + 1e-9

    def test_a_region_is_flagged_uncertain_when_one_sigma_changes_the_answer(self, engine):
        a = assess(engine, body=body(sigma=2.4))
        assert any(d.uncertain for s in a.sizes for d in s.regions)

    def test_nothing_acceptable_abstains_with_its_own_code(self, engine):
        a = assess(engine, body=body(waist=140.0, hip=160.0, sigma=0.5))
        assert a.recommendation.verdict is Verdict.ABSTAIN
        assert a.recommendation.abstain.code is AbstainCode.NO_SIZE_ACCEPTABLE


class TestRankingAndPreference:
    def test_a_bigger_body_is_recommended_a_bigger_size(self, engine):
        small = assess(engine, body=body(waist=76.0, hip=90.0, sigma=0.4, scale=0.003))
        large = assess(engine, body=body(waist=84.0, hip=98.0, sigma=0.4, scale=0.003))
        assert small.recommendation.primary.size_label < large.recommendation.primary.size_label

    def test_asking_for_looser_never_recommends_a_smaller_size(self, engine):
        tight = assess(engine, preference=FitPreference.TIGHTER)
        loose = assess(engine, preference=FitPreference.LOOSER)
        best = lambda a: max(a.sizes, key=lambda s: s.confidence).size_label
        assert best(loose) >= best(tight)

    def test_slim_intent_shifts_the_required_ease_down(self, engine):
        regular = assess(engine, garment=trousers(intent=FitIntent.REGULAR))
        slim = assess(engine, garment=trousers(intent=FitIntent.OVERSIZED))
        r = next(d for d in regular.sizes[0].regions if d.region is BodyRegion.WAIST)
        s = next(d for d in slim.sizes[0].regions if d.region is BodyRegion.WAIST)
        assert s.required_ease.preferred_cm > r.required_ease.preferred_cm

    def test_a_recommended_size_is_always_one_it_assessed(self, engine):
        a = assess(engine)
        labels = {s.size_label for s in a.sizes}
        assert a.recommendation.primary.size_label in labels


class TestCoverage:
    def test_a_region_the_garment_does_not_measure_is_declared_missing(self, engine):
        a = assess(engine, policy=policy(
            region_weights={BodyRegion.WAIST: 1.0, BodyRegion.HIP: 0.8, BodyRegion.THIGH: 0.4},
            critical_regions=frozenset({BodyRegion.WAIST}),
        ))
        assert a.sizes[0].coverage is Coverage.PARTIAL
        assert BodyRegion.THIGH in a.sizes[0].missing_regions

    def test_full_coverage_is_reported_as_complete(self, engine):
        a = assess(engine)
        assert a.sizes[0].coverage is Coverage.COMPLETE
        assert a.sizes[0].missing_regions == ()


class TestProperties:
    @settings(max_examples=40, deadline=None)
    @given(
        waist=st.floats(min_value=60, max_value=120),
        sigma=st.floats(min_value=0.3, max_value=2.4),
    )
    def test_confidences_always_sum_to_one(self, waist, sigma):
        from fitkit.fit_engine import DeterministicFitEngine

        a = assess(DeterministicFitEngine(), body=body(waist=waist, hip=waist + 15, sigma=sigma))
        assert sum(s.confidence for s in a.sizes) == pytest.approx(1.0, abs=1e-3)

    @settings(max_examples=30, deadline=None)
    @given(waist=st.floats(min_value=60, max_value=120))
    def test_every_document_is_serialisable_and_valid(self, waist):
        from fitkit.fit_engine import DeterministicFitEngine

        a = assess(DeterministicFitEngine(), body=body(waist=waist, hip=waist + 15))
        assert FitAssessment.from_dict(json.loads(json.dumps(a.to_dict()))) == a

    @settings(max_examples=30, deadline=None)
    @given(waist=st.floats(min_value=70, max_value=100), extra=st.floats(min_value=0.5, max_value=6))
    def test_a_larger_waist_never_makes_a_given_size_less_tight(self, waist, extra):
        from fitkit.fit_engine import DeterministicFitEngine

        e = DeterministicFitEngine()
        small = assess(e, body=body(waist=waist, hip=waist + 15))
        large = assess(e, body=body(waist=waist + extra, hip=waist + 15 + extra))
        pick = lambda a: next(
            d for d in next(s for s in a.sizes if s.size_label == "48").regions
            if d.region is BodyRegion.WAIST
        )
        assert pick(large).delta_cm <= pick(small).delta_cm + 1e-9


class TestStrategySeams:
    """Each strategy is swappable, which is the only thing that justifies it being one."""

    def test_a_region_with_no_ease_baseline_fails_loudly(self):
        from fitkit.fit_engine.ease import ConventionalEaseRules
        from fitkit.domain.regions import GarmentCategory

        with pytest.raises(KeyError, match="HEIGHT"):
            ConventionalEaseRules().required_ease(
                BodyRegion.HEIGHT, GarmentCategory.TROUSERS, FitIntent.REGULAR,
                FitPreference.AS_DESIGNED,
            )

    def test_lengths_do_not_stretch(self):
        from fitkit.fit_engine.stretch import ClassBasedStretchModel
        from fitkit.domain.regions import GarmentCategory

        model = ClassBasedStretchModel()
        assert model.usable_extension(JERSEY, BodyRegion.INSEAM, GarmentCategory.TROUSERS) == 0.0
        assert model.usable_extension(JERSEY, BodyRegion.WAIST, GarmentCategory.TROUSERS) > 0.0

    def test_unknown_recovery_is_discounted_but_not_zeroed(self):
        from fitkit.domain.fabric import FabricSpec, RecoveryClass, StretchClass
        from fitkit.domain.regions import GarmentCategory
        from fitkit.fit_engine.stretch import ClassBasedStretchModel

        model = ClassBasedStretchModel()
        unknown = FabricSpec(StretchClass.HIGH, RecoveryClass.UNKNOWN)
        good = model.usable_extension(JERSEY, BodyRegion.WAIST, GarmentCategory.TROUSERS)
        mid = model.usable_extension(unknown, BodyRegion.WAIST, GarmentCategory.TROUSERS)
        poor = model.usable_extension(BAGGY, BodyRegion.WAIST, GarmentCategory.TROUSERS)
        assert 0.0 < poor < mid < good

    def test_a_custom_abstain_policy_is_honoured(self):
        from fitkit.domain.contracts.fit_assessment import AbstainCode, Verdict
        from fitkit.fit_engine import DeterministicFitEngine
        from fitkit.fit_engine.abstain import Decision

        class NeverAnswers:
            policy_id = "abstain/never"

            def decide(self, ranked, body, policy):
                return Decision(Verdict.ABSTAIN, AbstainCode.NO_SIZE_ACCEPTABLE, ("policy",))

        a = assess(DeterministicFitEngine(abstain_policy=NeverAnswers()))
        assert a.recommendation.verdict is Verdict.ABSTAIN

    def test_the_engine_version_names_the_strategies_in_use(self, engine):
        version = engine.engine_version
        assert "ease/conventional/1" in version
        assert "stretch/class/1" in version
