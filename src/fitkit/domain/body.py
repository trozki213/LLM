"""Body measurements and the correlated-error model.

Per-region independent sigmas are wrong in the direction that matters. A mis-declared
height rescales *every* circumference the same way, which is a size shift; independent
noise smears them, which is a fit-quality question. Treating the first as the second
systematically under-abstains, so the shared component is modelled explicitly:

    sigma_total(r)^2 = (scale_sigma_rel * value(r))^2 + sigma_resid(r)^2

The `Measure` stored for each region carries the *total*, so any naive read
over-reports uncertainty rather than under-reporting it. The decomposition stays
available for the engine's quadrature via `residual_sigma_cm`.

Note: `residuals` holds the residual component of a sigma whose full value is
already stored as a Measure alongside it; wrapping the component again would imply it is
an independently meaningful quantity, which it is not.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from fitkit.domain.errors import MissingRegion
from fitkit.domain.regions import BodyRegion
from fitkit.domain.units import Measure

#: Sensitivity of a region to the shared scale factor. Circumferences and lengths are
#: linear in scale, so this is 1.0 for every region we currently model. A named constant
#: rather than a per-region table because one value is all the evidence supports; a table
#: with one repeated entry would be indirection without variation.
_SCALE_SENSITIVITY: float = 1.0

_TOTAL_SIGMA_TOLERANCE_CM: float = 1e-9


@dataclass(frozen=True, slots=True)
class MeasurementProvenance:
    """Everything needed to reproduce or attribute a measurement after the fact."""

    backend_id: str
    backend_version: str
    residual_table_version: str
    capture_id: str
    calibration_source_id: str
    computed_at: dt.datetime

    def __post_init__(self) -> None:
        if self.computed_at.tzinfo is None or self.computed_at.utcoffset() is None:
            raise ValueError(
                "computed_at must carry a timezone; naive timestamps are unattributable"
            )


@dataclass(frozen=True, slots=True)
class ScaleCalibration:
    """Metric scale and how well we know it. The C3 seam: height today, depth later."""

    source_id: str
    sigma_rel: float
    reference: Measure | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma_rel) or not 0.0 < self.sigma_rel < 1.0:
            raise ValueError(
                f"sigma_rel must be in (0, 1), got {self.sigma_rel!r}; scale is never exact "
                "and a relative error of 100% is not a calibration"
            )


@dataclass(frozen=True)
class BodyMeasurements:
    """A body, as far as we can tell, with the uncertainty we actually have.

    Only the independent quantities are stored: the per-region residual measures and the
    shared scale uncertainty. The totals are derived, so there is no way to construct an
    object whose stated sigma disagrees with its own components.
    """

    residuals: Mapping[BodyRegion, Measure]
    scale_sigma_rel: float
    provenance: MeasurementProvenance

    def __post_init__(self) -> None:
        _validate_scale(self.scale_sigma_rel)
        if not self.residuals:
            raise ValueError("a body must have at least one region measured")
        totals: dict[BodyRegion, Measure] = {}
        for region, measure in self.residuals.items():
            if measure.value_cm <= 0:
                raise ValueError(f"{region.name} must be positive, got {measure.value_cm!r} cm")
            totals[region] = Measure(
                value_cm=measure.value_cm,
                sigma_cm=_total_sigma(measure.value_cm, measure.sigma_cm, self.scale_sigma_rel),
                source=measure.source,
            )
        object.__setattr__(self, "residuals", MappingProxyType(dict(self.residuals)))
        object.__setattr__(self, "_totals", MappingProxyType(totals))

    @property
    def values(self) -> Mapping[BodyRegion, Measure]:
        """Per-region measures carrying *total* sigma, so a naive read is the safe read."""
        return self._totals  # type: ignore[attr-defined]

    def __getitem__(self, region: BodyRegion) -> Measure:
        try:
            return self.values[region]
        except KeyError:
            raise MissingRegion(region) from None

    def get(self, region: BodyRegion) -> Measure | None:
        return self.values.get(region)

    def residual_sigma_cm(self, region: BodyRegion) -> float:
        """The component of uncertainty that is independent of metric scale."""
        try:
            return self.residuals[region].sigma_cm
        except KeyError:
            raise MissingRegion(region) from None

    @property
    def regions(self) -> frozenset[BodyRegion]:
        return frozenset(self.residuals)


def _validate_scale(scale_sigma_rel: float) -> None:
    if not math.isfinite(scale_sigma_rel) or not 0.0 < scale_sigma_rel < 1.0:
        raise ValueError(
            f"scale_sigma_rel must be in (0, 1), got {scale_sigma_rel!r}; declared height "
            "is never exact, so a zero shared component is false precision"
        )


def _total_sigma(value_cm: float, residual_cm: float, scale_sigma_rel: float) -> float:
    return math.hypot(_SCALE_SENSITIVITY * scale_sigma_rel * value_cm, residual_cm)
