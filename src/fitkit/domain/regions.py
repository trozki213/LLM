"""The vocabulary of places on a body and on a garment, and how they correspond.

The correspondence is vocabulary, not arithmetic: this module says a garment's
`WAIST_FLAT` is about the body's `WAIST` and that it is a flat measure. Turning a flat
measure into a girth, and comparing the two, is the fit engine's job.
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType


class BodyRegion(StrEnum):
    HEIGHT = "height"
    BUST = "bust"
    UNDERBUST = "underbust"
    WAIST = "waist"
    HIP = "hip"
    THIGH = "thigh"
    NECK = "neck"
    SHOULDER_WIDTH = "shoulder_width"
    ARM_LENGTH = "arm_length"
    INSEAM = "inseam"


class GarmentCategory(StrEnum):
    TROUSERS = "trousers"
    SKIRT = "skirt"
    DRESS = "dress"
    TOP = "top"
    JACKET = "jacket"


class FitIntent(StrEnum):
    """The designer's intended ease, before the shopper's own preference is applied."""

    SLIM = "slim"
    REGULAR = "regular"
    OVERSIZED = "oversized"


class GarmentRegion(StrEnum):
    CHEST_FLAT = "chest_flat"
    WAIST_FLAT = "waist_flat"
    HIP_FLAT = "hip_flat"
    THIGH_FLAT = "thigh_flat"
    SHOULDER = "shoulder"
    SLEEVE_LENGTH = "sleeve_length"
    INSEAM = "inseam"

    @property
    def body_region(self) -> BodyRegion:
        """The body region this garment measurement is compared against."""
        return _BODY_COUNTERPART[self]

    @property
    def is_flat(self) -> bool:
        """True when the spec sheet measures the garment laid flat, i.e. half a girth."""
        return self in _FLAT_REGIONS


_BODY_COUNTERPART = MappingProxyType(
    {
        GarmentRegion.CHEST_FLAT: BodyRegion.BUST,
        GarmentRegion.WAIST_FLAT: BodyRegion.WAIST,
        GarmentRegion.HIP_FLAT: BodyRegion.HIP,
        GarmentRegion.THIGH_FLAT: BodyRegion.THIGH,
        GarmentRegion.SHOULDER: BodyRegion.SHOULDER_WIDTH,
        GarmentRegion.SLEEVE_LENGTH: BodyRegion.ARM_LENGTH,
        GarmentRegion.INSEAM: BodyRegion.INSEAM,
    }
)

_FLAT_REGIONS = frozenset(
    {
        GarmentRegion.CHEST_FLAT,
        GarmentRegion.WAIST_FLAT,
        GarmentRegion.HIP_FLAT,
        GarmentRegion.THIGH_FLAT,
    }
)
