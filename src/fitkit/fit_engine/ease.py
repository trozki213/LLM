"""How much room a garment needs, by region, category, intent and shopper preference."""

from __future__ import annotations

import typing

from fitkit.domain.policy import EaseWindow, FitPreference
from fitkit.domain.regions import BodyRegion, FitIntent, GarmentCategory

#: Baseline ease windows for a REGULAR fit, in centimetres of girth (or length).
#: Sourced from conventional pattern-cutting allowances, not from measured returns --
#: Phase 7 is what will replace them with fitted values.
_BASE = {
    BodyRegion.WAIST: (1.0, 2.5, 6.0),
    BodyRegion.HIP: (1.5, 3.5, 8.0),
    BodyRegion.BUST: (2.0, 5.0, 12.0),
    BodyRegion.UNDERBUST: (1.0, 2.5, 6.0),
    BodyRegion.THIGH: (2.0, 4.0, 9.0),
    BodyRegion.NECK: (0.5, 1.5, 4.0),
    BodyRegion.SHOULDER_WIDTH: (-0.5, 0.5, 2.0),
    BodyRegion.ARM_LENGTH: (-1.0, 0.0, 2.5),
    BodyRegion.INSEAM: (-1.0, 0.0, 3.0),
}

_INTENT_SHIFT_CM = {FitIntent.SLIM: -1.5, FitIntent.REGULAR: 0.0, FitIntent.OVERSIZED: 3.0}
_PREFERENCE_SHIFT_CM = {
    FitPreference.TIGHTER: -1.5,
    FitPreference.AS_DESIGNED: 0.0,
    FitPreference.LOOSER: 1.5,
}


class EaseRulePolicy(typing.Protocol):
    """Varies: category, brand fit philosophy, merchant override."""

    rules_id: str

    def required_ease(
        self,
        region: BodyRegion,
        category: GarmentCategory,
        intent: FitIntent,
        preference: FitPreference,
    ) -> EaseWindow: ...


class ConventionalEaseRules:
    """A shift-based model: one baseline per region, moved by intent and preference."""

    rules_id = "ease/conventional/1"

    def required_ease(
        self,
        region: BodyRegion,
        category: GarmentCategory,
        intent: FitIntent,
        preference: FitPreference,
    ) -> EaseWindow:
        try:
            low, pref, high = _BASE[region]
        except KeyError:
            raise KeyError(f"no ease baseline for {region.name}") from None
        shift = _INTENT_SHIFT_CM[intent] + _PREFERENCE_SHIFT_CM[preference]
        return EaseWindow(min_cm=low + shift, preferred_cm=pref + shift, max_cm=high + shift)
