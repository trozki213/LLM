"""Replay: a candidate engine re-run over historical assessments."""
import datetime as dt

import pytest

from fitkit.catalog import CsvSpecImporter, GarmentSpecBuilder, InMemoryGarmentRepository
from fitkit.domain.body import BodyMeasurements, MeasurementProvenance
from fitkit.domain.fabric import FabricSpec, RecoveryClass, StretchClass
from fitkit.domain.policy import FitPolicy, FitPreference
from fitkit.domain.regions import BodyRegion, FitIntent, GarmentCategory
from fitkit.domain.units import Measure, MeasureSource
from fitkit.evaluation import ReplayCase, replay
from fitkit.fit_engine import ConventionalEaseRules, DeterministicFitEngine
from fitkit.fit_engine.ease import _INTENT_SHIFT_CM  # noqa: F401  (documented below)

NOW = dt.datetime(2026, 9, 3, tzinfo=dt.UTC)
CSV = b"""size_label,region,value_cm,tolerance_cm
46,waist_flat,39.0,0.6
46,hip_flat,47.0,0.6
48,waist_flat,41.0,0.6
48,hip_flat,49.0,0.6
50,waist_flat,43.0,0.6
50,hip_flat,51.0,0.6
"""
POLICY = FitPolicy(
    policy_id="policy/merchant-a", version=3, tau_single=0.65, tau_pair=0.85,
    max_critical_sigma_cm=2.5,
    region_weights={BodyRegion.WAIST: 1.0, BodyRegion.HIP: 0.8},
    critical_regions=frozenset({BodyRegion.WAIST, BodyRegion.HIP}),
    tightness_penalty_ratio=1.8,
)


class ShiftedEaseRules:
    """A candidate policy that wants every garment two centimetres roomier."""

    rules_id = "ease/candidate/1"

    def required_ease(self, region, category, intent, preference):
        from fitkit.domain.policy import EaseWindow

        base = ConventionalEaseRules().required_ease(region, category, intent, preference)
        return EaseWindow(base.min_cm - 4.0, base.preferred_cm - 4.0, base.max_cm - 4.0)


def _garments():
    spec = (
        GarmentSpecBuilder()
        .with_identity("brand:sku-1", version=7, category=GarmentCategory.TROUSERS,
                       size_system="EU", fit_intent=FitIntent.REGULAR)
        .with_fabric(FabricSpec(StretchClass.NONE, RecoveryClass.GOOD))
        .with_grading_tolerance(0.6)
        .with_rows(CsvSpecImporter().parse(CSV).rows)
        .build()
    )
    return InMemoryGarmentRepository(spec)


def _body(waist=82.0):
    return BodyMeasurements(
        residuals={
            BodyRegion.WAIST: Measure(waist, 0.9, MeasureSource.ESTIMATED),
            BodyRegion.HIP: Measure(waist + 15, 0.9, MeasureSource.ESTIMATED),
        },
        scale_sigma_rel=0.005,
        provenance=MeasurementProvenance(
            "fixed", "1", "residuals/1", "cap_1", "declared-height", NOW
        ),
    )


def _cases(garments, engine, kept="48", n=3):
    cases = []
    for i in range(n):
        body = _body(80.0 + i)
        original = engine.assess(
            body=body, garment=garments.get("brand:sku-1"),
            preference=FitPreference.AS_DESIGNED, policy=POLICY,
            assessment_id=f"a{i}", computed_at=NOW,
        )
        cases.append(ReplayCase(original, body, FitPreference.AS_DESIGNED, kept))
    return cases


class TestReplay:
    def test_replaying_the_same_engine_changes_nothing(self):
        garments, engine = _garments(), DeterministicFitEngine()
        report = replay(_cases(garments, engine), engine=engine, garments=garments, policy=POLICY)
        assert report.changed == 0
        assert report.net == 0

    def test_a_candidate_policy_shows_up_as_changed_recommendations(self):
        garments, engine = _garments(), DeterministicFitEngine()
        candidate = DeterministicFitEngine(ease_rules=ShiftedEaseRules())
        report = replay(
            _cases(garments, engine), engine=candidate, garments=garments, policy=POLICY
        )
        assert report.changed > 0

    def test_it_reconstructs_the_exact_garment_version_that_was_used(self):
        """Replay is only valid because specs are immutable and versioned (ADR-009)."""
        garments, engine = _garments(), DeterministicFitEngine()
        cases = _cases(garments, engine)
        assert cases[0].original.inputs_digest.garment_spec_version == "brand:sku-1@7"
        replay(cases, engine=engine, garments=garments, policy=POLICY)

    def test_improvements_and_regressions_are_counted_against_what_was_kept(self):
        garments, engine = _garments(), DeterministicFitEngine()
        candidate = DeterministicFitEngine(ease_rules=ShiftedEaseRules())
        cases = _cases(garments, engine, kept="46")
        report = replay(cases, engine=candidate, garments=garments, policy=POLICY)
        assert report.improved + report.regressed <= len(cases)
        assert report.net == report.improved - report.regressed

    def test_an_outcome_with_no_kept_size_counts_neither_way(self):
        garments, engine = _garments(), DeterministicFitEngine()
        cases = _cases(garments, engine, kept=None)
        report = replay(cases, engine=engine, garments=garments, policy=POLICY)
        assert report.improved == 0 and report.regressed == 0
