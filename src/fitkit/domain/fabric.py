"""Fabric properties. C4: a 2 cm delta means different things on denim and on jersey."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class StretchClass(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RecoveryClass(StrEnum):
    """Whether the fabric returns to shape. Stretch without recovery is not usable ease."""

    GOOD = "good"
    POOR = "poor"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FabricSpec:
    """What we know about how the cloth behaves.

    `elongation_pct` must arrive with the load it was measured at. "35% stretch" with no
    stated force is not a physical quantity, and the type refuses to record one -- this
    is open question 11 encoded so it cannot be quietly skipped.
    """

    stretch_class: StretchClass
    recovery: RecoveryClass
    elongation_pct: float | None = None
    elongation_load_n: float | None = None
    composition: str | None = None

    def __post_init__(self) -> None:
        if self.elongation_pct is not None and self.elongation_load_n is None:
            raise ValueError(
                "elongation_load_n is required alongside elongation_pct: a stretch figure "
                "without the load it was measured at is not a measurable quantity"
            )
        if self.elongation_load_n is not None and self.elongation_pct is None:
            raise ValueError("elongation_pct is required alongside elongation_load_n")
        if self.elongation_pct is not None:
            if not math.isfinite(self.elongation_pct) or self.elongation_pct < 0:
                raise ValueError(f"elongation_pct must be finite and >= 0, got {self.elongation_pct!r}")
        if self.elongation_load_n is not None:
            if not math.isfinite(self.elongation_load_n) or self.elongation_load_n <= 0:
                raise ValueError(
                    f"elongation_load_n must be finite and > 0, got {self.elongation_load_n!r}"
                )
