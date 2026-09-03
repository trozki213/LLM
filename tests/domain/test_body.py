"""BodyMeasurements carries the correlated-error model, and refuses to under-report."""
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from fitkit.domain.body import BodyMeasurements, MeasurementProvenance
from fitkit.domain.errors import MissingRegion
from fitkit.domain.regions import BodyRegion
from fitkit.domain.units import Measure, MeasureSource

from tests.domain.factories import provenance, residual


def build(scale_rel: float = 0.01, **regions: float) -> BodyMeasurements:
    resid = {BodyRegion[k.upper()]: residual(v) for k, v in regions.items()}
    return BodyMeasurements(
        residuals=resid, scale_sigma_rel=scale_rel, provenance=provenance()
    )


class TestCorrelatedErrorModel:
    def test_stored_sigma_is_the_total_not_the_residual(self):
        """A naive read must over-report uncertainty, never under-report it."""
        body = build(scale_rel=0.02, waist=80.0)
        total = body[BodyRegion.WAIST].sigma_cm
        expected = math.hypot(0.02 * 80.0, residual(80.0).sigma_cm)
        assert total == pytest.approx(expected)
        assert total > body.residual_sigma_cm(BodyRegion.WAIST)

    def test_residual_component_is_still_available_for_the_engine(self):
        body = build(scale_rel=0.02, waist=80.0)
        assert body.residual_sigma_cm(BodyRegion.WAIST) == pytest.approx(residual(80.0).sigma_cm)

    def test_scale_component_scales_with_the_measurement(self):
        small = build(scale_rel=0.02, waist=60.0)
        large = build(scale_rel=0.02, waist=120.0)
        assert large[BodyRegion.WAIST].sigma_cm > small[BodyRegion.WAIST].sigma_cm

    def test_zero_scale_uncertainty_is_rejected(self):
        """Declared height is never exact; a zero shared component is false precision."""
        with pytest.raises(ValueError, match="scale_sigma_rel"):
            build(scale_rel=0.0, waist=80.0)

    @pytest.mark.parametrize("bad", [-0.01, 1.0, 2.0])
    def test_implausible_scale_uncertainty_is_rejected(self, bad):
        with pytest.raises(ValueError, match="scale_sigma_rel"):
            build(scale_rel=bad, waist=80.0)


class TestContainerInvariants:
    def test_rejects_non_positive_circumference(self):
        with pytest.raises(ValueError, match="WAIST"):
            build(waist=-5.0)

    def test_rejects_an_empty_body(self):
        with pytest.raises(ValueError, match="at least one region"):
            build()

    def test_missing_region_raises_a_typed_error(self):
        body = build(waist=80.0)
        with pytest.raises(MissingRegion) as exc:
            body[BodyRegion.HIP]
        assert exc.value.region is BodyRegion.HIP

    def test_get_returns_none_for_a_missing_region(self):
        assert build(waist=80.0).get(BodyRegion.HIP) is None

    def test_regions_reports_what_is_present(self):
        body = build(waist=80.0, hip=95.0)
        assert body.regions == frozenset({BodyRegion.WAIST, BodyRegion.HIP})

    def test_is_defensive_against_caller_mutation(self):
        source = {BodyRegion.WAIST: residual(80.0)}
        body = BodyMeasurements(
            residuals=source, scale_sigma_rel=0.01, provenance=provenance()
        )
        source[BodyRegion.HIP] = residual(95.0)
        assert body.regions == frozenset({BodyRegion.WAIST})

    def test_exposed_mapping_cannot_be_mutated(self):
        body = build(waist=80.0)
        with pytest.raises(TypeError):
            body.values[BodyRegion.HIP] = residual(95.0)  # type: ignore[index]


class TestProvenance:
    def test_provenance_is_required(self):
        with pytest.raises(TypeError):
            BodyMeasurements(  # type: ignore[call-arg]
                residuals={BodyRegion.WAIST: residual(80.0)}, scale_sigma_rel=0.01
            )

    def test_timestamp_must_be_timezone_aware(self):
        import datetime as dt

        with pytest.raises(ValueError, match="timezone"):
            MeasurementProvenance(
                backend_id="fake",
                backend_version="1",
                residual_table_version="1",
                capture_id="c1",
                calibration_source_id="declared-height",
                computed_at=dt.datetime(2026, 9, 3, 12, 0, 0),
            )


class TestProperties:
    @given(
        value=st.floats(min_value=20, max_value=200, allow_nan=False),
        resid=st.floats(min_value=0.05, max_value=10, allow_nan=False),
        scale=st.floats(min_value=0.001, max_value=0.2, allow_nan=False),
    )
    def test_total_sigma_always_dominates_both_components(self, value, resid, scale):
        body = BodyMeasurements(
            residuals={BodyRegion.WAIST: Measure(value, resid, MeasureSource.ESTIMATED)},
            scale_sigma_rel=scale,
            provenance=provenance(),
        )
        total = body[BodyRegion.WAIST].sigma_cm
        assert total >= resid - 1e-9
        assert total >= scale * value - 1e-9


class TestScaleCalibration:
    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, float("nan")])
    def test_rejects_an_implausible_relative_sigma(self, bad):
        from fitkit.domain.body import ScaleCalibration

        with pytest.raises(ValueError, match="sigma_rel"):
            ScaleCalibration(source_id="declared-height", sigma_rel=bad)

    def test_carries_the_reference_it_was_derived_from(self):
        from fitkit.domain.body import ScaleCalibration

        ref = Measure(175.0, 1.5, MeasureSource.USER_DECLARED)
        cal = ScaleCalibration(source_id="declared-height", sigma_rel=0.009, reference=ref)
        assert cal.reference is ref


class TestTotalsAreDerivedNotStored:
    def test_the_total_cannot_disagree_with_its_components(self):
        """There is no constructor parameter for the total, so it cannot be wrong."""
        import dataclasses

        fields = {f.name for f in dataclasses.fields(BodyMeasurements)}
        assert fields == {"residuals", "scale_sigma_rel", "provenance"}

    def test_equality_ignores_the_derived_totals(self):
        a, b = build(waist=80.0), build(waist=80.0)
        assert a == b

    def test_residual_lookup_of_a_missing_region_is_typed(self):
        body = build(waist=80.0)
        with pytest.raises(MissingRegion):
            body.residual_sigma_cm(BodyRegion.HIP)
