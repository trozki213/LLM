"""Builders for engine tests. The engine is pure, so no doubles are needed anywhere."""

from __future__ import annotations

import datetime as dt

import pytest

from fitkit.domain.body import BodyMeasurements, MeasurementProvenance
from fitkit.domain.fabric import FabricSpec, RecoveryClass, StretchClass
from fitkit.domain.garment import GarmentSizeSpec, GarmentSpec
from fitkit.domain.policy import FitPolicy
from fitkit.domain.regions import BodyRegion, FitIntent, GarmentCategory, GarmentRegion
from fitkit.domain.units import Measure, MeasureSource

FIXED_TIME = dt.datetime(2026, 9, 3, 11, 4, 22, tzinfo=dt.UTC)

RIGID = FabricSpec(StretchClass.NONE, RecoveryClass.GOOD)
JERSEY = FabricSpec(StretchClass.HIGH, RecoveryClass.GOOD)
BAGGY = FabricSpec(StretchClass.HIGH, RecoveryClass.POOR)


def body(*, waist: float = 80.0, hip: float = 95.0, sigma: float = 1.2, scale: float = 0.009):
    return BodyMeasurements(
        residuals={
            BodyRegion.WAIST: Measure(waist, sigma, MeasureSource.ESTIMATED),
            BodyRegion.HIP: Measure(hip, sigma, MeasureSource.ESTIMATED),
        },
        scale_sigma_rel=scale,
        provenance=MeasurementProvenance(
            backend_id="fixed",
            backend_version="1",
            residual_table_version="residuals/1",
            capture_id="cap_01J",
            calibration_source_id="declared-height",
            computed_at=FIXED_TIME,
        ),
    )


def trousers(
    *,
    fabric: FabricSpec = RIGID,
    intent: FitIntent = FitIntent.REGULAR,
    waists: tuple[float, ...] = (39.0, 41.0, 43.0),
    labels: tuple[str, ...] = ("46", "48", "50"),
    hip_offset: float = 8.0,
    tolerance: float = 0.5,
    version: int = 7,
) -> GarmentSpec:
    """Flat measures, so the girth is twice these numbers. Steps of 2 cm flat = 4 cm girth."""
    sizes = tuple(
        GarmentSizeSpec(
            size_label=label,
            measurements={
                GarmentRegion.WAIST_FLAT: Measure(w, tolerance, MeasureSource.SPEC_SHEET),
                GarmentRegion.HIP_FLAT: Measure(w + hip_offset, tolerance, MeasureSource.SPEC_SHEET),
            },
        )
        for label, w in zip(labels, waists, strict=True)
    )
    return GarmentSpec(
        garment_id="brand:sku-1",
        version=version,
        category=GarmentCategory.TROUSERS,
        size_system="EU",
        fit_intent=intent,
        fabric=fabric,
        sizes=sizes,
    )


def policy(**overrides) -> FitPolicy:
    fields = dict(
        policy_id="policy/merchant-a",
        version=3,
        tau_single=0.65,
        tau_pair=0.85,
        max_critical_sigma_cm=2.5,
        region_weights={BodyRegion.WAIST: 1.0, BodyRegion.HIP: 0.8},
        critical_regions=frozenset({BodyRegion.WAIST, BodyRegion.HIP}),
        tightness_penalty_ratio=1.8,
    )
    fields.update(overrides)
    return FitPolicy(**fields)


@pytest.fixture
def engine():
    from fitkit.fit_engine import DeterministicFitEngine

    return DeterministicFitEngine()


def assess(engine, **kwargs):
    from fitkit.domain.policy import FitPreference

    params = dict(
        body=body(),
        garment=trousers(),
        preference=FitPreference.AS_DESIGNED,
        policy=policy(),
        assessment_id="01JBQ7H3M4N5P6Q7R8S9T0V1W2",
        computed_at=FIXED_TIME,
    )
    params.update(kwargs)
    return engine.assess(**params)
