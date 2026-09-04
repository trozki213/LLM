"""Offline replay of a candidate engine or policy over historical assessments.

Replay is only valid because garment specs are immutable and versioned and the contract
carries an `inputs_digest` (ADR-007, ADR-009). It gives a cheap offline signal; the
randomised holdout gives the truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from fitkit.domain.body import BodyMeasurements
from fitkit.domain.contracts.fit_assessment import FitAssessment, Verdict
from fitkit.domain.policy import FitPolicy, FitPreference
from fitkit.domain.ports import GarmentRepository
from fitkit.fit_engine.engine import DeterministicFitEngine


@dataclass(frozen=True, slots=True)
class ReplayCase:
    """A historical assessment plus the body that produced it and what was kept."""

    original: FitAssessment
    body: BodyMeasurements
    preference: FitPreference
    kept_size: str | None


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    assessment_id: str
    before: str | None
    after: str | None
    kept_size: str | None

    @property
    def changed(self) -> bool:
        return self.before != self.after

    @property
    def improved(self) -> bool:
        return self.kept_size is not None and self.after == self.kept_size != self.before

    @property
    def regressed(self) -> bool:
        return self.kept_size is not None and self.before == self.kept_size != self.after


@dataclass(frozen=True, slots=True)
class ReplayReport:
    outcomes: tuple[ReplayOutcome, ...]

    @property
    def changed(self) -> int:
        return sum(1 for o in self.outcomes if o.changed)

    @property
    def improved(self) -> int:
        return sum(1 for o in self.outcomes if o.improved)

    @property
    def regressed(self) -> int:
        return sum(1 for o in self.outcomes if o.regressed)

    @property
    def net(self) -> int:
        return self.improved - self.regressed


def replay(
    cases: Iterable[ReplayCase],
    *,
    engine: DeterministicFitEngine,
    garments: GarmentRepository,
    policy: FitPolicy,
) -> ReplayReport:
    outcomes = []
    for case in cases:
        garment_id, _, version = case.original.inputs_digest.garment_spec_version.rpartition("@")
        garment = garments.get(garment_id, int(version))
        candidate = engine.assess(
            body=case.body,
            garment=garment,
            preference=case.preference,
            policy=policy,
            assessment_id=case.original.assessment_id,
            computed_at=case.original.inputs_digest.computed_at,
            locale=case.original.render_hints.locale,
            tone=case.original.render_hints.tone,
        )
        outcomes.append(
            ReplayOutcome(
                assessment_id=case.original.assessment_id,
                before=_chosen(case.original),
                after=_chosen(candidate),
                kept_size=case.kept_size,
            )
        )
    return ReplayReport(tuple(outcomes))


def _chosen(assessment: FitAssessment) -> str | None:
    rec = assessment.recommendation
    return None if rec.verdict is Verdict.ABSTAIN else rec.primary.size_label
