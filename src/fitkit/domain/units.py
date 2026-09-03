"""Quantities that cannot exist without an uncertainty.

`Measure` is the load-bearing type of the whole system. C6 says silent false precision
must be impossible; the way to make something impossible is to make it unconstructible,
so there is no way to express a length in this codebase without also expressing how well
we know it.

bare-cm-exempt: this module *defines* the wrapper, so it is the one place where a
centimetre is stored as a plain float.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

#: Below these, a claimed uncertainty is indistinguishable from a claim of certainty.
MIN_SIGMA_CM: Final[float] = 0.01
MIN_SIGMA_KG: Final[float] = 0.01


class MeasureSource(StrEnum):
    """Where a quantity came from. Recorded so provenance survives arithmetic."""

    ESTIMATED = "estimated"
    USER_DECLARED = "user_declared"
    TAPE = "tape"
    SPEC_SHEET = "spec_sheet"
    DERIVED = "derived"


@dataclass(frozen=True, slots=True)
class Measure:
    """A length in centimetres with a mandatory 1-sigma uncertainty.

    Values may be negative: a fit delta ("waist -2 cm") is a Measure too. Domain
    positivity is a *container* invariant, enforced by whoever holds the measure, not
    by the value type -- see `BodyMeasurements` and `GarmentSizeSpec`.
    """

    value_cm: float
    sigma_cm: float
    source: MeasureSource

    def __post_init__(self) -> None:
        _validate_uncertain(
            self.value_cm, self.sigma_cm, self.source, "value_cm", "sigma_cm", MIN_SIGMA_CM, "cm"
        )

    def __sub__(self, other: Measure) -> Measure:
        """Difference of two *independent* measurements.

        Independence holds for the case this exists to serve -- a garment measurement
        minus a body measurement. It does not hold between two regions of the same body,
        whose shared scale error is modelled by `BodyMeasurements`, not here.
        """
        if not isinstance(other, Measure):
            return NotImplemented
        return Measure(
            value_cm=self.value_cm - other.value_cm,
            sigma_cm=math.hypot(self.sigma_cm, other.sigma_cm),
            source=MeasureSource.DERIVED,
        )

    def __add__(self, other: Measure) -> Measure:
        """Sum of two independent measurements. Same independence caveat as `__sub__`."""
        if not isinstance(other, Measure):
            return NotImplemented
        return Measure(
            value_cm=self.value_cm + other.value_cm,
            sigma_cm=math.hypot(self.sigma_cm, other.sigma_cm),
            source=MeasureSource.DERIVED,
        )

    def scaled(self, factor: float) -> Measure:
        """Scale by an exactly-known factor, e.g. doubling a flat measure to a girth.

        Sigma scales with the value, but is clamped at `MIN_SIGMA_CM`: shrinking a
        quantity must not shrink our uncertainty below the point where we would be
        claiming certainty. The clamp always over-reports, never under-reports.
        """
        if not math.isfinite(factor) or factor == 0.0:
            raise ValueError(
                f"factor must be finite and non-zero, got {factor!r}; scaling to zero would "
                "manufacture certainty"
            )
        return Measure(
            value_cm=self.value_cm * factor,
            sigma_cm=max(abs(factor) * self.sigma_cm, MIN_SIGMA_CM),
            source=MeasureSource.DERIVED,
        )


@dataclass(frozen=True, slots=True)
class Mass:
    """A mass in kilograms with a mandatory 1-sigma uncertainty.

    Weight is dimensionally not a length, so it does not travel in a `Measure`. It exists
    because the published evidence makes declared weight the single strongest input to
    measurement accuracy (design 7.5), so the seam should already be the right shape when
    the product decision is made -- not because anything reads it today.
    """

    value_kg: float
    sigma_kg: float
    source: MeasureSource

    def __post_init__(self) -> None:
        _validate_uncertain(
            self.value_kg, self.sigma_kg, self.source, "value_kg", "sigma_kg", MIN_SIGMA_KG, "kg"
        )


def _validate_uncertain(
    value: float,
    sigma: float,
    source: object,
    value_name: str,
    sigma_name: str,
    min_sigma: float,
    unit: str,
) -> None:
    """The one rule both quantity types share: no value without a real uncertainty."""
    if not isinstance(source, MeasureSource):
        raise TypeError(f"source must be a MeasureSource, got {type(source).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{value_name} must be finite, got {value!r}")
    if not math.isfinite(sigma) or sigma < min_sigma:
        raise ValueError(
            f"{sigma_name} must be finite and at least {min_sigma} {unit}, got {sigma!r}. "
            "Every measurement carries uncertainty; there is no exact value here."
        )
