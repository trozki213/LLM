"""How much of a garment's circumference is actually available to the wearer.

C4 lives here. A 2 cm shortfall on rigid denim and on elastane jersey are different
outcomes, and the difference is expressed as *usable extension*: the fraction of the
garment's measurement a wearer can take up without the garment feeling wrong.
"""

from __future__ import annotations

import typing

from fitkit.domain.fabric import FabricSpec, RecoveryClass, StretchClass
from fitkit.domain.regions import BodyRegion, GarmentCategory

#: Usable extension by stretch class. Deliberately far below the fabric's elongation at
#: break: this is what a wearer accepts in normal wear, not what the cloth can survive.
#: These are placeholders pending the measurement convention in open question 11, and
#: Phase 7 is what will replace them with fitted values.
_CLASS_EXTENSION = {
    StretchClass.NONE: 0.0,
    StretchClass.LOW: 0.03,
    StretchClass.MEDIUM: 0.08,
    StretchClass.HIGH: 0.15,
}

#: Poor recovery means the garment stretches and stays stretched. That is not usable
#: ease, it is a garment that bags out, so only a token allowance survives.
_POOR_RECOVERY_FACTOR = 0.25
_UNKNOWN_RECOVERY_FACTOR = 0.5

#: Lengths do not stretch usefully in wear the way girths do.
_LENGTH_REGIONS = frozenset({BodyRegion.INSEAM, BodyRegion.ARM_LENGTH, BodyRegion.SHOULDER_WIDTH})


class StretchModel(typing.Protocol):
    """Varies: how usable extension is derived. Class-based now, tension curve later."""

    model_id: str

    def usable_extension(
        self, fabric: FabricSpec, region: BodyRegion, category: GarmentCategory
    ) -> float: ...


class ClassBasedStretchModel:
    """v1: stretch class, discounted by recovery. No elongation figure is required."""

    model_id = "stretch/class/1"

    def usable_extension(
        self, fabric: FabricSpec, region: BodyRegion, category: GarmentCategory
    ) -> float:
        if region in _LENGTH_REGIONS:
            return 0.0
        base = _CLASS_EXTENSION[fabric.stretch_class]
        if fabric.recovery is RecoveryClass.POOR:
            return base * _POOR_RECOVERY_FACTOR
        if fabric.recovery is RecoveryClass.UNKNOWN:
            return base * _UNKNOWN_RECOVERY_FACTOR
        return base
