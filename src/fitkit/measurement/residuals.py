"""Measured residuals, and the decorator that attaches them.

ADR-010: our uncertainty estimate is a measurement, not an opinion. A vendor returning
bare point estimates would otherwise get a sigma chosen by whoever wrote the adapter,
and that number would become folklore. Instead it comes from a versioned table that
Phase 7 builds from the tape-measured validation panel.
"""

from __future__ import annotations

from dataclasses import dataclass

from fitkit.domain.body import BodyMeasurements, ScaleCalibration
from fitkit.domain.capture import CaptureBundle
from fitkit.domain.errors import UncalibratedBackend
from fitkit.domain.ports import MeasurementBackend
from fitkit.domain.regions import BodyRegion
from fitkit.domain.units import Measure


@dataclass(frozen=True, slots=True)
class ResidualEntry:
    """The measured residual for one backend, region and body-shape bucket.

    Buckets exist because error concentrates: the published evidence shows measurement
    error rising sharply at high BMI, and a single global figure would quietly promise
    the average shopper's accuracy to everyone.
    """

    backend_id: str
    region: BodyRegion
    upper_bound_cm: float
    residual_cm: float

    def __post_init__(self) -> None:
        if self.residual_cm <= 0:
            raise ValueError(
                f"residual_cm must be > 0 for {self.backend_id}/{self.region.name}, got "
                f"{self.residual_cm!r}"
            )


@dataclass(frozen=True, slots=True)
class ResidualTable:
    version: str
    entries: tuple[ResidualEntry, ...]

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("a residual table must be versioned; it is recorded in every assessment")
        object.__setattr__(self, "entries", tuple(sorted(
            self.entries, key=lambda e: (e.backend_id, e.region.name, e.upper_bound_cm)
        )))

    def residual_cm(self, backend_id: str, region: BodyRegion, value_cm: float) -> float:
        candidates = [
            e for e in self.entries if e.backend_id == backend_id and e.region is region
        ]
        if not candidates:
            raise UncalibratedBackend(
                f"no measured residual for {backend_id}/{region.name}; characterise the "
                "backend on the validation panel before shipping it"
            )
        for entry in candidates:
            if value_cm <= entry.upper_bound_cm:
                return entry.residual_cm
        return candidates[-1].residual_cm


class UncertaintyCalibrator:
    """Decorator. Varies where sigma comes from, independently of where the value does.

    Without it a black-box vendor silently gets sigma = 0 or a guessed constant, and C6
    is defeated at the one seam where we have least visibility.
    """

    def __init__(self, inner: MeasurementBackend, table: ResidualTable) -> None:
        self._inner = inner
        self._table = table

    @property
    def backend_id(self) -> str:
        return self._inner.backend_id

    @property
    def supported_regions(self) -> frozenset[BodyRegion]:
        return self._inner.supported_regions

    def estimate(self, bundle: CaptureBundle, calibration: ScaleCalibration) -> BodyMeasurements:
        raw = self._inner.estimate(bundle, calibration)
        residuals = {
            region: Measure(
                value_cm=raw.values[region].value_cm,
                sigma_cm=self._table.residual_cm(
                    self._inner.backend_id, region, raw.values[region].value_cm
                ),
                source=raw.values[region].source,
            )
            for region in sorted(raw.regions, key=lambda r: r.name)
        }
        provenance = type(raw.provenance)(
            backend_id=raw.provenance.backend_id,
            backend_version=raw.provenance.backend_version,
            residual_table_version=self._table.version,
            capture_id=raw.provenance.capture_id,
            calibration_source_id=raw.provenance.calibration_source_id,
            computed_at=raw.provenance.computed_at,
        )
        return BodyMeasurements(
            residuals=residuals,
            scale_sigma_rel=calibration.sigma_rel,
            provenance=provenance,
        )
