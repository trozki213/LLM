"""Measure is the type that makes C6 structural: no length exists without uncertainty."""
import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from fitkit.domain.units import MIN_SIGMA_CM, Measure, MeasureSource

TAPE = MeasureSource.TAPE
EST = MeasureSource.ESTIMATED


def m(value: float, sigma: float = 1.0, source: MeasureSource = EST) -> Measure:
    return Measure(value_cm=value, sigma_cm=sigma, source=source)


class TestConstruction:
    def test_exposes_its_fields(self):
        x = m(82.0, 1.4, TAPE)
        assert x.value_cm == 82.0
        assert x.sigma_cm == 1.4
        assert x.source is TAPE

    def test_rejects_zero_sigma(self):
        """The headline C6 rule: you cannot claim a length you are certain of."""
        with pytest.raises(ValueError, match="sigma_cm"):
            m(82.0, 0.0)

    def test_rejects_negative_sigma(self):
        with pytest.raises(ValueError, match="sigma_cm"):
            m(82.0, -0.5)

    def test_rejects_sigma_below_floor(self):
        with pytest.raises(ValueError, match="sigma_cm"):
            m(82.0, MIN_SIGMA_CM / 2)

    def test_accepts_sigma_at_the_floor(self):
        assert m(82.0, MIN_SIGMA_CM).sigma_cm == MIN_SIGMA_CM

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_value(self, bad):
        with pytest.raises(ValueError, match="value_cm"):
            m(bad, 1.0)

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_rejects_non_finite_sigma(self, bad):
        with pytest.raises(ValueError, match="sigma_cm"):
            m(82.0, bad)

    def test_allows_negative_values(self):
        """Deltas are Measures too. Positivity is a container invariant, not a value one."""
        assert m(-2.0, 1.4).value_cm == -2.0

    def test_requires_a_source(self):
        with pytest.raises(TypeError):
            Measure(value_cm=82.0, sigma_cm=1.0)  # type: ignore[call-arg]


class TestImmutability:
    def test_is_frozen(self):
        x = m(82.0)
        with pytest.raises(AttributeError):
            x.value_cm = 90.0  # type: ignore[misc]

    def test_equality_is_by_value(self):
        assert m(82.0, 1.4, TAPE) == m(82.0, 1.4, TAPE)
        assert m(82.0, 1.4, TAPE) != m(82.0, 1.4, EST)

    def test_is_hashable(self):
        assert len({m(82.0, 1.4, TAPE), m(82.0, 1.4, TAPE)}) == 1


class TestArithmetic:
    def test_subtraction_combines_sigma_in_quadrature(self):
        d = m(80.0, 3.0) - m(78.0, 4.0)
        assert d.value_cm == pytest.approx(2.0)
        assert d.sigma_cm == pytest.approx(5.0)  # sqrt(9 + 16)

    def test_addition_combines_sigma_in_quadrature(self):
        s = m(80.0, 3.0) + m(78.0, 4.0)
        assert s.value_cm == pytest.approx(158.0)
        assert s.sigma_cm == pytest.approx(5.0)

    def test_arithmetic_marks_the_result_as_derived(self):
        assert (m(80.0, 1.0, TAPE) - m(78.0, 1.0, TAPE)).source is MeasureSource.DERIVED

    def test_scaled_scales_value_and_sigma(self):
        x = m(40.0, 0.5).scaled(2.0)  # flat measure to circumference
        assert x.value_cm == pytest.approx(80.0)
        assert x.sigma_cm == pytest.approx(1.0)

    def test_scaled_by_a_negative_factor_keeps_sigma_positive(self):
        x = m(40.0, 0.5).scaled(-2.0)
        assert x.value_cm == pytest.approx(-80.0)
        assert x.sigma_cm == pytest.approx(1.0)

    def test_shrinking_clamps_sigma_at_the_floor(self):
        """Scaling down must not scale our certainty up past what we actually know."""
        assert m(40.0, 0.02).scaled(0.1).sigma_cm == MIN_SIGMA_CM

    def test_scaling_by_zero_is_rejected(self):
        """Scaling to zero would manufacture certainty out of nothing."""
        with pytest.raises(ValueError, match="factor"):
            m(40.0, 0.5).scaled(0.0)

    def test_cannot_subtract_a_bare_float(self):
        with pytest.raises(TypeError):
            m(80.0, 1.0) - 2.0  # type: ignore[operator]


finite = st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False)
sigmas = st.floats(min_value=MIN_SIGMA_CM, max_value=50, allow_nan=False, allow_infinity=False)
measures = st.builds(Measure, value_cm=finite, sigma_cm=sigmas, source=st.just(EST))


class TestProperties:
    @given(a=measures, b=measures)
    def test_subtraction_never_reduces_uncertainty(self, a, b):
        assert (a - b).sigma_cm >= max(a.sigma_cm, b.sigma_cm) - 1e-9

    @given(a=measures, b=measures)
    def test_subtraction_is_antisymmetric_in_value_and_symmetric_in_sigma(self, a, b):
        fwd, rev = a - b, b - a
        assert fwd.value_cm == pytest.approx(-rev.value_cm)
        assert fwd.sigma_cm == pytest.approx(rev.sigma_cm)

    @given(a=measures, k=st.floats(min_value=0.01, max_value=100, allow_nan=False))
    def test_scaling_is_linear_in_sigma_above_the_floor(self, a, k):
        expected = max(abs(k) * a.sigma_cm, MIN_SIGMA_CM)
        assert a.scaled(k).sigma_cm == pytest.approx(expected)

    @given(a=measures, k=st.floats(min_value=0.01, max_value=100, allow_nan=False))
    def test_scaling_never_reports_less_uncertainty_than_the_floor(self, a, k):
        assert a.scaled(k).sigma_cm >= MIN_SIGMA_CM

    @given(a=measures, b=measures)
    def test_every_arithmetic_result_is_a_valid_measure(self, a, b):
        for r in (a - b, a + b, a.scaled(3.0)):
            assert math.isfinite(r.value_cm)
            assert r.sigma_cm >= MIN_SIGMA_CM


class TestMass:
    """Weight is not a length. It gets its own type, with the same uncertainty rule."""

    def test_carries_kilograms_and_their_uncertainty(self):
        from fitkit.domain.units import Mass

        w = Mass(72.0, 2.0, MeasureSource.USER_DECLARED)
        assert (w.value_kg, w.sigma_kg) == (72.0, 2.0)

    def test_rejects_zero_uncertainty(self):
        from fitkit.domain.units import Mass

        with pytest.raises(ValueError, match="sigma_kg"):
            Mass(72.0, 0.0, MeasureSource.USER_DECLARED)

    def test_has_no_centimetre_field(self):
        from fitkit.domain.units import Mass

        assert not hasattr(Mass(72.0, 2.0, MeasureSource.USER_DECLARED), "value_cm")

    def test_is_not_interchangeable_with_a_measure(self):
        from fitkit.domain.units import Mass

        assert Mass(72.0, 2.0, TAPE) != Measure(72.0, 2.0, TAPE)


class TestArithmeticTypeSafety:
    def test_addition_rejects_a_bare_float(self):
        with pytest.raises(TypeError):
            m(80.0, 1.0) + 2.0  # type: ignore[operator]

    def test_a_measure_cannot_absorb_a_mass(self):
        from fitkit.domain.units import Mass

        with pytest.raises(TypeError):
            m(80.0, 1.0) - Mass(72.0, 2.0, EST)  # type: ignore[operator]


class TestMassValidation:
    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_rejects_non_finite_mass(self, bad):
        from fitkit.domain.units import Mass

        with pytest.raises(ValueError, match="value_kg"):
            Mass(bad, 2.0, EST)

    def test_rejects_a_source_that_is_not_a_measure_source(self):
        from fitkit.domain.units import Mass

        with pytest.raises(TypeError, match="MeasureSource"):
            Mass(72.0, 2.0, "user_declared")  # type: ignore[arg-type]

    def test_measure_also_rejects_a_bad_source(self):
        with pytest.raises(TypeError, match="MeasureSource"):
            Measure(80.0, 1.0, "tape")  # type: ignore[arg-type]
