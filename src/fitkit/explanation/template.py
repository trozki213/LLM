"""The reference renderer.

ADR-008: this is not a fallback bolted on afterwards. It is written first, it defines
what correct output means, and the LLM is measured against it. Because it is exactly
reproducible it can be golden-file tested, and because it is free and needs no network
the system can ship without an LLM at all.
"""

from __future__ import annotations

from fitkit.domain.contracts.explanation import Explanation
from fitkit.domain.contracts.fit_assessment import (
    Coverage,
    FitAssessment,
    RegionDelta,
    SizeAssessment,
    Verdict,
)
from fitkit.explanation.phrases import (
    CLASSIFICATION_PHRASES,
    REGION_NAMES,
    language_of,
)

_TEXT = {
    "en": {
        "single": "Order the {size}.",
        "two_sizes": "Either the {primary} or the {alternate} should work.",
        "abstain": "We can't confidently call a size for you on this one.",
        "abstain_reason": {
            "UNCERTAINTY_EXCEEDS_SIZE_STEP": (
                "Your measurements are close enough to the boundary between sizes that "
                "we'd only be guessing."
            ),
            "NO_SIZE_ACCEPTABLE": "No size in this run comes close enough to fit you.",
            "INSUFFICIENT_GARMENT_DATA": "We don't have enough measurements for this garment.",
            "INSUFFICIENT_BODY_DATA": "We don't have enough of your measurements yet.",
        },
        "region": "{region}: {phrase}, by about {amount} cm.",
        "region_exact": "{region}: {phrase}.",
        "uncertain": "Treat the {regions} figure as approximate.",
        "uncertain_plural": "Treat the {regions} figures as approximate.",
        "partial": "We couldn't check the {regions} on this garment.",
        "stretch": "The fabric takes up about {amount} cm of that at the {region}.",
        "confidence": "Confidence: {pct}%.",
    },
    "it": {
        "single": "Ordina la {size}.",
        "two_sizes": "Vanno bene sia la {primary} sia la {alternate}.",
        "abstain": "Non possiamo indicarti una taglia con sufficiente sicurezza.",
        "abstain_reason": {
            "UNCERTAINTY_EXCEEDS_SIZE_STEP": (
                "Le tue misure sono cosi vicine al confine tra due taglie che sarebbe "
                "solo un'ipotesi."
            ),
            "NO_SIZE_ACCEPTABLE": "Nessuna taglia di questa serie ti sta abbastanza bene.",
            "INSUFFICIENT_GARMENT_DATA": "Non abbiamo abbastanza misure di questo capo.",
            "INSUFFICIENT_BODY_DATA": "Non abbiamo ancora abbastanza tue misure.",
        },
        "region": "{region}: {phrase}, di circa {amount} cm.",
        "region_exact": "{region}: {phrase}.",
        "uncertain": "Considera approssimativo il dato su {regions}.",
        "uncertain_plural": "Considera approssimativi i dati su {regions}.",
        "partial": "Non abbiamo potuto verificare {regions} su questo capo.",
        "stretch": "Il tessuto assorbe circa {amount} cm di questo scarto su {region}.",
        "confidence": "Affidabilita: {pct}%.",
    },
}


class TemplateRenderer:
    renderer_id = "template/1"

    def render(self, assessment: FitAssessment) -> Explanation:
        lang = language_of(assessment.render_hints.locale)
        text = _TEXT[lang]
        lines: list[str] = []
        rec = assessment.recommendation

        if rec.verdict is Verdict.ABSTAIN:
            lines.append(text["abstain"])
            reason = text["abstain_reason"].get(rec.abstain.code.value)
            if reason:
                lines.append(reason)
            focus = _focus_size(assessment)
        else:
            if rec.verdict is Verdict.SINGLE:
                lines.append(text["single"].format(size=rec.primary.size_label))
            else:
                lines.append(
                    text["two_sizes"].format(
                        primary=rec.primary.size_label, alternate=rec.alternate.size_label
                    )
                )
            lines.append(text["confidence"].format(pct=_pct(rec.primary.confidence)))
            focus = _size_named(assessment, rec.primary.size_label)

        if focus is not None:
            lines.extend(self._region_lines(focus, lang))
            lines.extend(self._caveats(focus, lang))

        return Explanation(
            text=" ".join(lines), renderer_id=self.renderer_id, degraded=False, notes=(lang,)
        )

    def _region_lines(self, size: SizeAssessment, lang: str) -> list[str]:
        text = _TEXT[lang]
        out = []
        for delta in sorted(size.regions, key=lambda d: (not d.critical, d.region.name)):
            name = REGION_NAMES[lang][delta.region.value]
            phrase = CLASSIFICATION_PHRASES[lang][delta.classification]
            amount = abs(delta.delta_cm)
            if amount < 0.05:
                out.append(text["region_exact"].format(region=name.capitalize(), phrase=phrase))
            else:
                out.append(
                    text["region"].format(
                        region=name.capitalize(), phrase=phrase, amount=_num(amount)
                    )
                )
            if delta.stretch_absorbed_cm >= 0.05:
                out.append(
                    text["stretch"].format(
                        amount=_num(delta.stretch_absorbed_cm), region=name
                    )
                )
        return out

    def _caveats(self, size: SizeAssessment, lang: str) -> list[str]:
        text = _TEXT[lang]
        out = []
        uncertain = [
            REGION_NAMES[lang][d.region.value] for d in size.regions if d.uncertain
        ]
        if uncertain:
            key = "uncertain" if len(uncertain) == 1 else "uncertain_plural"
            out.append(text[key].format(regions=_join(uncertain, lang)))
        if size.coverage is Coverage.PARTIAL and size.missing_regions:
            names = [REGION_NAMES[lang][r.value] for r in size.missing_regions]
            out.append(text["partial"].format(regions=_join(names, lang)))
        return out


def _focus_size(assessment: FitAssessment) -> SizeAssessment | None:
    """When abstaining, still describe the closest size -- silence explains nothing."""
    with_regions = [s for s in assessment.sizes if s.regions]
    if not with_regions:
        return None
    return max(with_regions, key=lambda s: (s.confidence, s.size_label))


def _size_named(assessment: FitAssessment, label: str) -> SizeAssessment | None:
    return next((s for s in assessment.sizes if s.size_label == label), None)


def _num(value: float) -> str:
    """Render a centimetre figure the way the allowlist represents it."""
    rounded = round(value, 1)
    return str(int(rounded)) if rounded == int(rounded) else f"{rounded:.1f}"


def _pct(confidence: float) -> str:
    return str(int(round(confidence * 100)))


def _join(names: list[str], lang: str) -> str:
    conjunction = " and " if lang == "en" else " e "
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + conjunction + names[-1]
