"""What the harness reads. Both record types come from outside the system."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fitkit.domain.regions import BodyRegion


@dataclass(frozen=True, slots=True)
class GroundTruthSample:
    """One region of one subject, as we estimated it and as the tape measured it."""

    capture_id: str
    backend_id: str
    region: BodyRegion
    estimated_cm: float
    estimated_sigma_cm: float
    tape_cm: float

    @property
    def error_cm(self) -> float:
        return self.estimated_cm - self.tape_cm

    @property
    def within_one_sigma(self) -> bool:
        return abs(self.error_cm) <= self.estimated_sigma_cm


class Arm(StrEnum):
    TREATMENT = "treatment"
    CONTROL = "control"


class ReturnReason(StrEnum):
    KEPT = "kept"
    TOO_SMALL = "too_small"
    TOO_BIG = "too_big"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class OutcomeRecord:
    """One order, joined to the assessment that preceded it by `assessment_id`."""

    assessment_id: str
    arm: Arm
    verdict: str
    recommended_size: str | None
    confidence: float | None
    purchased_size: str
    return_reason: ReturnReason

    @property
    def size_related_return(self) -> bool:
        return self.return_reason in (ReturnReason.TOO_SMALL, ReturnReason.TOO_BIG)

    @property
    def followed_recommendation(self) -> bool:
        return self.recommended_size is not None and self.purchased_size == self.recommended_size
