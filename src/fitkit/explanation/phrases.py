"""The controlled vocabulary.

C1 protects the arithmetic; this protects the rhetoric. An LLM that faithfully reports
"waist -2 cm" and then writes "should still feel comfortable" has changed the purchase
decision without touching a number, so what may be *said* about each classification is
enumerated here and enforced by a guard.
"""

from __future__ import annotations

from types import MappingProxyType

from fitkit.domain.contracts.fit_assessment import FitClassification as F

DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = ("en", "it")

REGION_NAMES = MappingProxyType({
    "en": {
        "bust": "bust", "underbust": "underbust", "waist": "waist", "hip": "hip",
        "thigh": "thigh", "neck": "neck", "shoulder_width": "shoulders",
        "arm_length": "sleeve length", "inseam": "inseam", "height": "height",
    },
    "it": {
        "bust": "petto", "underbust": "sottoseno", "waist": "vita", "hip": "fianchi",
        "thigh": "coscia", "neck": "collo", "shoulder_width": "spalle",
        "arm_length": "lunghezza manica", "inseam": "cavallo", "height": "altezza",
    },
})

CLASSIFICATION_PHRASES = MappingProxyType({
    "en": {
        F.MUCH_TOO_TIGHT: "far too tight",
        F.TIGHT: "tighter than this cut is meant to sit",
        F.SNUG: "close-fitting",
        F.AS_INTENDED: "sitting as the designer intended",
        F.RELAXED: "a little relaxed",
        F.LOOSE: "looser than this cut is meant to sit",
        F.MUCH_TOO_LOOSE: "far too loose",
    },
    "it": {
        F.MUCH_TOO_TIGHT: "decisamente troppo stretto",
        F.TIGHT: "piu stretto di quanto previsto da questo modello",
        F.SNUG: "aderente",
        F.AS_INTENDED: "come previsto dal modello",
        F.RELAXED: "leggermente morbido",
        F.LOOSE: "piu largo di quanto previsto da questo modello",
        F.MUCH_TOO_LOOSE: "decisamente troppo largo",
    },
})

#: Words that contradict a tight classification. Saying any of these about a region the
#: engine called tight is a guard violation, whatever the numbers alongside it say.
REASSURANCE_TERMS = MappingProxyType({
    "en": ("roomy", "plenty of room", "loose", "generous", "relaxed", "spacious", "forgiving"),
    "it": ("comodo", "abbondante", "largo", "morbido", "generoso"),
})

#: Words that contradict a loose classification.
CONSTRICTION_TERMS = MappingProxyType({
    "en": ("tight", "snug", "close-fitting", "restrictive", "fitted"),
    "it": ("stretto", "aderente", "attillato"),
})

TIGHT_CLASSES = frozenset({F.MUCH_TOO_TIGHT, F.TIGHT})
LOOSE_CLASSES = frozenset({F.MUCH_TOO_LOOSE, F.LOOSE})


def language_of(locale: str) -> str:
    """`it-IT` -> `it`, with a documented fallback rather than a silent failure."""
    code = locale.split("-")[0].split("_")[0].lower()
    return code if code in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
