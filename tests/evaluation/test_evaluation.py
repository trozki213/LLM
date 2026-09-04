"""Evaluation: honest sigmas, outcome metrics against a control, and replay."""
import datetime as dt

import pytest

from fitkit.domain.regions import BodyRegion
from fitkit.evaluation import (
    ACCEPTABLE_COVERAGE,
    Arm,
    GroundTruthSample,
    OutcomeRecord,
    ReturnReason,
    build_residual_table,
    compare_arms,
    measurement_accuracy,
    outcome_report,
    risk_coverage_curve,
)
from fitkit.evaluation.residual_fitting import MIN_SAMPLES_PER_BUCKET
from fitkit.measurement.residuals import UncertaintyCalibrator


def samples(n=20, region=BodyRegion.WAIST, error=1.0, sigma=1.5, base=80.0):
    return [
        GroundTruthSample(
            capture_id=f"cap_{i}", backend_id="vendor", region=region,
            estimated_cm=base + (error if i % 2 else -error),
            estimated_sigma_cm=sigma, tape_cm=base,
        )
        for i in range(n)
    ]


def outcomes(n=100, arm=Arm.TREATMENT, return_rate=0.2, verdict="SINGLE"):
    return [
        OutcomeRecord(
            assessment_id=f"a{i}", arm=arm, verdict=verdict,
            recommended_size="48", confidence=0.5 + (i % 5) / 10,
            purchased_size="48",
            return_reason=ReturnReason.TOO_SMALL if i < int(n * return_rate) else ReturnReason.KEPT,
        )
        for i in range(n)
    ]


class TestMeasurementAccuracy:
    def test_reports_error_per_region(self):
        report = measurement_accuracy(samples(error=1.0))
        waist = report[BodyRegion.WAIST]
        assert waist.mae_cm == pytest.approx(1.0)
        assert waist.rmse_cm == pytest.approx(1.0)
        assert waist.samples == 20

    def test_detects_a_systematic_bias(self):
        biased = [
            GroundTruthSample(f"c{i}", "vendor", BodyRegion.WAIST, 82.0, 1.5, 80.0)
            for i in range(20)
        ]
        assert measurement_accuracy(biased)[BodyRegion.WAIST].bias_cm == pytest.approx(2.0)

    def test_an_honest_sigma_is_recognised(self):
        report = measurement_accuracy(samples(error=1.0, sigma=1.5))
        assert report[BodyRegion.WAIST].sigma_coverage == 1.0
        assert not report[BodyRegion.WAIST].sigma_is_honest  # over-covering is also dishonest

    def test_an_overconfident_sigma_is_caught(self):
        """The check that makes C6 real rather than decorative."""
        report = measurement_accuracy(samples(error=3.0, sigma=0.5))
        assert report[BodyRegion.WAIST].sigma_coverage == 0.0
        assert not report[BodyRegion.WAIST].sigma_is_honest

    def test_a_well_calibrated_sigma_passes(self):
        mixed = samples(n=100, error=1.0, sigma=1.5)[:68] + [
            GroundTruthSample(f"x{i}", "vendor", BodyRegion.WAIST, 84.0, 1.5, 80.0)
            for i in range(32)
        ]
        coverage = measurement_accuracy(mixed)[BodyRegion.WAIST].sigma_coverage
        assert ACCEPTABLE_COVERAGE[0] <= coverage <= ACCEPTABLE_COVERAGE[1]
        assert measurement_accuracy(mixed)[BodyRegion.WAIST].sigma_is_honest


class TestResidualFitting:
    def test_produces_a_table_the_runtime_can_consume(self):
        table = build_residual_table(samples(error=1.4), version="residuals/test")
        assert table.version == "residuals/test"
        assert table.residual_cm("vendor", BodyRegion.WAIST, 80.0) == pytest.approx(1.4, abs=0.01)

    def test_the_fitted_table_plugs_straight_into_the_calibrator(self):
        from tests.fakes import FixedMeasurementBackend

        table = build_residual_table(
            [
                GroundTruthSample(f"c{i}", "fixed", BodyRegion.WAIST, 80.0 + (1 if i % 2 else -1), 1.0, 80.0)
                for i in range(20)
            ]
            + [
                GroundTruthSample(f"h{i}", "fixed", BodyRegion.HIP, 95.0 + (1 if i % 2 else -1), 1.0, 95.0)
                for i in range(20)
            ],
            version="residuals/fitted",
        )
        calibrated = UncertaintyCalibrator(FixedMeasurementBackend(), table)
        assert calibrated.backend_id == "fixed"

    def test_a_panel_too_small_to_characterise_refuses_to_guess(self):
        with pytest.raises(ValueError, match="too small"):
            build_residual_table(samples(n=MIN_SAMPLES_PER_BUCKET - 1), version="v")

    def test_buckets_are_fitted_separately(self):
        small = samples(n=20, base=70.0, error=1.0)
        large = samples(n=20, base=110.0, error=3.0)
        table = build_residual_table(small + large, version="v")
        assert table.residual_cm("vendor", BodyRegion.WAIST, 70.0) < table.residual_cm(
            "vendor", BodyRegion.WAIST, 110.0
        )


class TestOutcomeMetrics:
    def test_reports_the_size_related_return_rate(self):
        report = outcome_report(outcomes(return_rate=0.25))
        assert report.size_related_return_rate == pytest.approx(0.25)
        assert report.too_small_rate == pytest.approx(0.25)

    def test_reports_abstention_and_coverage(self):
        mixed = outcomes(n=50) + outcomes(n=50, verdict="ABSTAIN")
        report = outcome_report(mixed)
        assert report.abstention_rate == pytest.approx(0.5)
        assert report.coverage == pytest.approx(0.5)

    def test_an_empty_set_is_refused(self):
        with pytest.raises(ValueError, match="empty"):
            outcome_report([])


class TestRandomisedComparison:
    def test_reports_the_reduction_against_the_control(self):
        records = outcomes(n=100, arm=Arm.TREATMENT, return_rate=0.15) + outcomes(
            n=100, arm=Arm.CONTROL, return_rate=0.30
        )
        comparison = compare_arms(records)
        assert comparison.absolute_reduction == pytest.approx(0.15)
        assert comparison.relative_reduction == pytest.approx(0.5)

    def test_refuses_to_report_without_a_control_arm(self):
        """ADR-011: an observational comparison here would flatter us."""
        with pytest.raises(ValueError, match="randomised control"):
            compare_arms(outcomes(arm=Arm.TREATMENT))


class TestRiskCoverage:
    def test_raising_the_threshold_reduces_coverage(self):
        curve = risk_coverage_curve(outcomes(n=100))
        coverages = [p.coverage for p in curve]
        assert coverages == sorted(coverages, reverse=True)

    def test_abstained_orders_never_count_as_answered(self):
        curve = risk_coverage_curve(outcomes(n=100, verdict="ABSTAIN"))
        assert all(p.coverage == 0.0 for p in curve)

    def test_the_curve_is_refused_on_an_empty_set(self):
        with pytest.raises(ValueError):
            risk_coverage_curve([])
