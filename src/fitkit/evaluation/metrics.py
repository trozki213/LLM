"""The two questions the harness exists to answer.

7a: are our measurements and our sigmas honest?
7b: does the recommendation reduce size-related returns?

The second is the metric that ultimately matters, and it is only trustworthy against a
randomised control (ADR-011): a recommendation changes what people buy, so comparing
recommended orders with unrecommended ones flatters us.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from fitkit.domain.regions import BodyRegion
from fitkit.evaluation.records import Arm, GroundTruthSample, OutcomeRecord

#: Nominal coverage of a one-sigma interval for a normal error distribution, and the
#: band within which we accept it. Outside this, abstention is decorative and C6 is
#: not actually being met.
NOMINAL_COVERAGE = 0.68
ACCEPTABLE_COVERAGE = (0.60, 0.76)


@dataclass(frozen=True, slots=True)
class RegionAccuracy:
    region: BodyRegion
    samples: int
    mae_cm: float
    bias_cm: float
    rmse_cm: float
    sigma_coverage: float

    @property
    def sigma_is_honest(self) -> bool:
        low, high = ACCEPTABLE_COVERAGE
        return low <= self.sigma_coverage <= high


def measurement_accuracy(samples: Iterable[GroundTruthSample]) -> dict[BodyRegion, RegionAccuracy]:
    grouped: dict[BodyRegion, list[GroundTruthSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.region].append(sample)

    report = {}
    for region, group in grouped.items():
        errors = [s.error_cm for s in group]
        report[region] = RegionAccuracy(
            region=region,
            samples=len(group),
            mae_cm=sum(abs(e) for e in errors) / len(errors),
            bias_cm=sum(errors) / len(errors),
            rmse_cm=math.sqrt(sum(e * e for e in errors) / len(errors)),
            sigma_coverage=sum(1 for s in group if s.within_one_sigma) / len(group),
        )
    return report


@dataclass(frozen=True, slots=True)
class OutcomeReport:
    orders: int
    abstention_rate: float
    coverage: float
    followed_rate: float
    size_related_return_rate: float
    too_small_rate: float
    too_big_rate: float


def outcome_report(records: Sequence[OutcomeRecord]) -> OutcomeReport:
    if not records:
        raise ValueError("cannot report on an empty set of outcomes")
    n = len(records)
    abstained = sum(1 for r in records if r.verdict == "ABSTAIN")
    return OutcomeReport(
        orders=n,
        abstention_rate=abstained / n,
        coverage=1.0 - abstained / n,
        followed_rate=sum(1 for r in records if r.followed_recommendation) / n,
        size_related_return_rate=sum(1 for r in records if r.size_related_return) / n,
        too_small_rate=sum(1 for r in records if r.return_reason.value == "too_small") / n,
        too_big_rate=sum(1 for r in records if r.return_reason.value == "too_big") / n,
    )


@dataclass(frozen=True, slots=True)
class ArmComparison:
    treatment: OutcomeReport
    control: OutcomeReport

    @property
    def absolute_reduction(self) -> float:
        return self.control.size_related_return_rate - self.treatment.size_related_return_rate

    @property
    def relative_reduction(self) -> float:
        """The headline number, framed as relative because the absolute ceiling is bounded
        by a ~2 cm measurement error against a ~4 cm size step."""
        base = self.control.size_related_return_rate
        return self.absolute_reduction / base if base else 0.0


def compare_arms(records: Sequence[OutcomeRecord]) -> ArmComparison:
    treatment = [r for r in records if r.arm is Arm.TREATMENT]
    control = [r for r in records if r.arm is Arm.CONTROL]
    if not treatment or not control:
        raise ValueError(
            "both arms are required; without a randomised control this comparison is "
            "confounded by the recommendation changing what people buy (ADR-011)"
        )
    return ArmComparison(outcome_report(treatment), outcome_report(control))


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    threshold: float
    coverage: float
    size_related_return_rate: float


def risk_coverage_curve(
    records: Sequence[OutcomeRecord], thresholds: Sequence[float] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)
) -> tuple[RiskCoveragePoint, ...]:
    """Accuracy against how often we are willing to answer.

    Without this, accuracy alone is meaningless: a system that abstains on everything is
    perfectly accurate and worthless, and one that never abstains hides its failures in
    the return rate.
    """
    answered = [r for r in records if r.confidence is not None and r.verdict != "ABSTAIN"]
    total = len(records)
    if not total:
        raise ValueError("cannot build a risk-coverage curve from no outcomes")

    points = []
    for threshold in thresholds:
        kept = [r for r in answered if r.confidence >= threshold]
        rate = (
            sum(1 for r in kept if r.size_related_return) / len(kept) if kept else 0.0
        )
        points.append(RiskCoveragePoint(threshold, len(kept) / total, rate))
    return tuple(points)
