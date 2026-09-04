"""The teeth behind C1.

Without these, "the LLM never decides the size" is a hope about a prompt. With them it
is a property of the system: anything a renderer says that the document does not support
is caught after generation and replaced by the template's output.
"""

from __future__ import annotations

import re
import typing
from dataclasses import dataclass

from fitkit.domain.contracts.fit_assessment import FitAssessment, Verdict
from fitkit.explanation.phrases import (
    CONSTRICTION_TERMS,
    LOOSE_CLASSES,
    REASSURANCE_TERMS,
    TIGHT_CLASSES,
    language_of,
)

#: Matches 12, 12.5 and 12,5. Thousands separators are out of range for the quantities
#: this contract carries, so the ambiguity does not arise in practice.
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

DEFAULT_MAX_CHARS = 700


@dataclass(frozen=True, slots=True)
class Violation:
    guard_id: str
    code: str
    detail: str


class Guard(typing.Protocol):
    guard_id: str

    def check(self, assessment: FitAssessment, text: str) -> tuple[Violation, ...]: ...


class NumericGuard:
    """Every numeral in the output must be one the document actually states (R4)."""

    guard_id = "numeric"

    def check(self, assessment: FitAssessment, text: str) -> tuple[Violation, ...]:
        allowed = assessment.numeric_allowlist()  # hoisted: built once, not per numeral
        tolerance = 0.005
        violations = []
        for token in _NUMBER.findall(text):
            value = float(token.replace(",", "."))
            if not any(abs(value - a) <= tolerance for a in allowed):
                violations.append(
                    Violation(self.guard_id, "unsupported_number", token)
                )
        return tuple(violations)


class BannedClaimGuard:
    """Words that contradict the engine's own classification.

    This is the guard the numeric one cannot replace: "should still be comfortable" on a
    -2 cm waist in rigid denim contains no numeral at all.
    """

    guard_id = "banned_claim"

    def check(self, assessment: FitAssessment, text: str) -> tuple[Violation, ...]:
        lang = language_of(assessment.render_hints.locale)
        lowered = text.lower()
        classes = {
            d.classification
            for size in assessment.sizes
            for d in size.regions
            if d.critical and _is_described(assessment, size.size_label)
        }
        violations = []
        if classes & TIGHT_CLASSES:
            violations += [
                Violation(self.guard_id, "reassurance_over_tight_fit", term)
                for term in REASSURANCE_TERMS[lang]
                if term in lowered
            ]
        if classes & LOOSE_CLASSES:
            violations += [
                Violation(self.guard_id, "constriction_over_loose_fit", term)
                for term in CONSTRICTION_TERMS[lang]
                if term in lowered
            ]
        return tuple(violations)


class AbstentionGuard:
    """When the engine abstained, no size may be named. Abstaining and then naming a
    size is the exact failure C1 exists to prevent, expressed in prose."""

    guard_id = "abstention"

    def check(self, assessment: FitAssessment, text: str) -> tuple[Violation, ...]:
        if assessment.recommendation.verdict is not Verdict.ABSTAIN:
            return ()
        lowered = text.lower()
        return tuple(
            Violation(self.guard_id, "size_named_while_abstaining", size.size_label)
            for size in assessment.sizes
            if re.search(_size_pattern(size.size_label), lowered)
        )


class LengthGuard:
    guard_id = "length"

    def __init__(self, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        self._max = max_chars

    def check(self, assessment: FitAssessment, text: str) -> tuple[Violation, ...]:
        if len(text) <= self._max:
            return ()
        return (Violation(self.guard_id, "too_long", f"{len(text)} > {self._max}"),)


def default_guards() -> tuple[Guard, ...]:
    return (NumericGuard(), BannedClaimGuard(), AbstentionGuard(), LengthGuard())


def _is_described(assessment: FitAssessment, size_label: str) -> bool:
    """Only the size actually being talked about constrains the language used."""
    rec = assessment.recommendation
    if rec.verdict is Verdict.ABSTAIN:
        described = max(assessment.sizes, key=lambda s: (s.confidence, s.size_label))
        return size_label == described.size_label
    return size_label in {
        c.size_label for c in (rec.primary, rec.alternate) if c is not None
    }


def _size_pattern(size_label: str) -> str:
    """Match a size label as a whole token.

    The trailing guard rejects `48.5` but must still match `48` at the end of a
    sentence, so it excludes a following decimal digit rather than a following period.
    """
    escaped = re.escape(size_label.lower())
    return rf"(?<![\w.]){escaped}(?!\d)(?!\.\d)"
