"""Shared builders for domain tests. Deliberately dumb: no logic worth testing lives here."""
from __future__ import annotations

import datetime as dt

from fitkit.domain.body import MeasurementProvenance
from fitkit.domain.units import Measure, MeasureSource

FIXED_TIME = dt.datetime(2026, 9, 3, 11, 4, 22, tzinfo=dt.UTC)


def residual(value: float, sigma: float = 1.2) -> Measure:
    return Measure(value_cm=value, sigma_cm=sigma, source=MeasureSource.ESTIMATED)


def spec(value: float, sigma: float = 0.5) -> Measure:
    return Measure(value_cm=value, sigma_cm=sigma, source=MeasureSource.SPEC_SHEET)


def provenance(**overrides) -> MeasurementProvenance:
    fields = dict(
        backend_id="fake-backend",
        backend_version="2026.07",
        residual_table_version="residuals/1",
        capture_id="cap_01J",
        calibration_source_id="declared-height",
        computed_at=FIXED_TIME,
    )
    fields.update(overrides)
    return MeasurementProvenance(**fields)


def assessment(**overrides):
    """A complete, valid FitAssessment. The reference fixture for Phases 4 and 5."""
    from fitkit.domain.contracts.fit_assessment import (
        AbstainReason,
        Coverage,
        FabricSummary,
        FitAssessment,
        FitClassification,
        GarmentRef,
        Recommendation,
        RegionDelta,
        RenderHints,
        SizeAssessment,
        SizeChoice,
        Verdict,
    )
    from fitkit.domain.fabric import RecoveryClass, StretchClass
    from fitkit.domain.policy import EaseWindow, Tone
    from fitkit.domain.regions import BodyRegion, FitIntent, GarmentCategory

    waist = RegionDelta(
        region=BodyRegion.WAIST,
        critical=True,
        delta_cm=-2.0,
        delta_sigma_cm=1.4,
        stretch_absorbed_cm=1.2,
        required_ease=EaseWindow(min_cm=1.0, preferred_cm=2.0, max_cm=5.0),
        classification=FitClassification.TIGHT,
        uncertain=False,
    )
    hip = RegionDelta(
        region=BodyRegion.HIP,
        critical=True,
        delta_cm=1.0,
        delta_sigma_cm=1.6,
        stretch_absorbed_cm=0.0,
        required_ease=EaseWindow(min_cm=0.5, preferred_cm=2.5, max_cm=6.0),
        classification=FitClassification.AS_INTENDED,
        uncertain=False,
    )
    fields = dict(
        assessment_id="01JBQ7H3M4N5P6Q7R8S9T0V1W2",
        garment=GarmentRef(
            garment_id="brand:sku-1",
            category=GarmentCategory.TROUSERS,
            size_system="EU",
            fit_intent=FitIntent.REGULAR,
        ),
        fabric=FabricSummary(
            stretch_class=StretchClass.LOW,
            recovery=RecoveryClass.GOOD,
            usable_extension_pct=6.0,
        ),
        recommendation=Recommendation(
            verdict=Verdict.SINGLE,
            primary=SizeChoice(size_label="48", confidence=0.71),
            alternate=None,
            abstain=None,
        ),
        sizes=(
            SizeAssessment(
                size_label="48",
                confidence=0.71,
                regions=(waist, hip),
                coverage=Coverage.COMPLETE,
                missing_regions=(),
            ),
        ),
        inputs_digest=digest(),
        render_hints=RenderHints(locale="it-IT", tone=Tone.NEUTRAL),
    )
    fields.update(overrides)
    return FitAssessment(**fields)


def digest(**overrides):
    from fitkit.domain.contracts.fit_assessment import InputsDigest

    fields = dict(
        measurement_backend="fake-backend@2026.07",
        measurement_provenance_id="cap_01J",
        garment_spec_version="brand:sku-1@7",
        engine_version="fit-engine/1.4.2",
        policy_version="policy/merchant-a/3",
        residual_table_version="residuals/1",
        computed_at=FIXED_TIME,
    )
    fields.update(overrides)
    return InputsDigest(**fields)
