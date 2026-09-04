"""When to refuse to answer.

Abstention is a first-class outcome (C6), and its thresholds are merchant policy rather
than engine constants: a retailer with free returns tolerates a confident guess that a
made-to-order brand cannot.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from fitkit.domain.contracts.fit_assessment import (
    AbstainCode,
    FitClassification,
    SizeAssessment,
    Verdict,
)
from fitkit.domain.body import BodyMeasurements
from fitkit.domain.policy import FitPolicy
from fitkit.domain.regions import BodyRegion

_INTOLERABLE = frozenset({FitClassification.MUCH_TOO_TIGHT, FitClassification.MUCH_TOO_LOOSE})


@dataclass(frozen=True, slots=True)
class RankedSize:
    assessment: SizeAssessment
    probability: float


@dataclass(frozen=True, slots=True)
class Decision:
    verdict: Verdict
    abstain_code: AbstainCode | None
    detail_codes: tuple[str, ...]


class AbstainPolicy(typing.Protocol):
    """Varies: merchant risk appetite."""

    policy_id: str

    def decide(
        self, ranked: tuple[RankedSize, ...], body: BodyMeasurements, policy: FitPolicy
    ) -> Decision: ...


class ThresholdAbstainPolicy:
    """Refuse when the measurement is too coarse, when nothing fits, or when we cannot tell."""

    policy_id = "abstain/threshold/1"

    def decide(
        self, ranked: tuple[RankedSize, ...], body: BodyMeasurements, policy: FitPolicy
    ) -> Decision:
        noisy = self._noisy_critical_regions(body, policy)
        if noisy:
            return Decision(
                Verdict.ABSTAIN,
                AbstainCode.UNCERTAINTY_EXCEEDS_SIZE_STEP,
                tuple(f"{r.name.lower()}_sigma_over_ceiling" for r in noisy),
            )
        if not ranked:
            return Decision(Verdict.ABSTAIN, AbstainCode.INSUFFICIENT_GARMENT_DATA, ())

        top = ranked[0]
        intolerable = tuple(
            d.region.name.lower()
            for d in top.assessment.regions
            if d.critical and d.classification in _INTOLERABLE
        )
        if intolerable:
            return Decision(
                Verdict.ABSTAIN,
                AbstainCode.NO_SIZE_ACCEPTABLE,
                tuple(f"{name}_out_of_range" for name in intolerable),
            )
        if top.probability >= policy.tau_single:
            return Decision(Verdict.SINGLE, None, ())
        if len(ranked) >= 2 and top.probability + ranked[1].probability >= policy.tau_pair:
            return Decision(Verdict.TWO_SIZES, None, ())
        return Decision(
            Verdict.ABSTAIN,
            AbstainCode.UNCERTAINTY_EXCEEDS_SIZE_STEP,
            ("no_size_reaches_confidence_threshold",),
        )

    @staticmethod
    def _noisy_critical_regions(
        body: BodyMeasurements, policy: FitPolicy
    ) -> tuple[BodyRegion, ...]:
        return tuple(
            sorted(
                (
                    region
                    for region in policy.critical_regions
                    if (m := body.get(region)) is not None
                    and m.sigma_cm > policy.max_critical_sigma_cm
                ),
                key=lambda r: r.name,
            )
        )
