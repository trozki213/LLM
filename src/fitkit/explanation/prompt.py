"""Prompt construction. The model receives the document and a job description, nothing else."""

from __future__ import annotations

import json

from fitkit.domain.contracts.fit_assessment import FitAssessment
from fitkit.explanation.phrases import CLASSIFICATION_PHRASES, language_of

_INSTRUCTIONS = """You turn a fit assessment into two or three short sentences for a shopper.

Rules, all of them absolute:
- Every number you write must already appear in the JSON below. Do not compute, convert,
  average or round anything into a number that is not there.
- Do not choose or suggest a size. The size decision is already made, in `recommendation`.
- If `verdict` is ABSTAIN, do not name any size at all.
- Describe each region using the meaning given for its `classification`. Do not soften a
  tight fit or firm up a loose one.
- Write in {language}. Plain, direct, no marketing language, no greeting, no sign-off.

Meanings of the classifications you may see:
{glossary}

Assessment:
{document}
"""


def build_prompt(assessment: FitAssessment) -> str:
    lang = language_of(assessment.render_hints.locale)
    glossary = "\n".join(
        f"- {key.value}: {phrase}" for key, phrase in CLASSIFICATION_PHRASES[lang].items()
    )
    return _INSTRUCTIONS.format(
        language={"en": "English", "it": "Italian"}[lang],
        glossary=glossary,
        document=json.dumps(assessment.to_dict(), sort_keys=True, indent=2),
    )
