"""Garment specifications: real physical measurements, with their grading tolerance.

Design 7.1: manufacturing tolerance is of the same order as our body-measurement error
and is invisible to us. So a garment measurement is a `Measure` like any other, and a
spec with zero tolerance is as much false precision as a body measurement with none.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from fitkit.domain.errors import SizeNotFound, SizeSpecIncomplete
from fitkit.domain.fabric import FabricSpec
from fitkit.domain.regions import FitIntent, GarmentCategory, GarmentRegion
from fitkit.domain.units import Measure


@dataclass(frozen=True, slots=True)
class GarmentSizeSpec:
    size_label: str
    measurements: Mapping[GarmentRegion, Measure]

    def __post_init__(self) -> None:
        if not self.size_label.strip():
            raise ValueError("size_label must not be blank")
        if not self.measurements:
            raise ValueError("a size must have at least one measurement")
        for region, measure in self.measurements.items():
            if measure.value_cm <= 0:
                raise ValueError(f"{region.name} must be positive, got {measure.value_cm!r} cm")
        object.__setattr__(self, "measurements", MappingProxyType(dict(self.measurements)))

    @property
    def regions(self) -> frozenset[GarmentRegion]:
        return frozenset(self.measurements)


@dataclass(frozen=True, slots=True)
class GarmentSpec:
    """One garment at one immutable version. Updates create a new version (ADR-009)."""

    garment_id: str
    version: int
    category: GarmentCategory
    size_system: str
    fit_intent: FitIntent
    fabric: FabricSpec
    sizes: tuple[GarmentSizeSpec, ...]

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError(f"version must be >= 1, got {self.version!r}")
        if not self.sizes:
            raise ValueError("a garment must have at least one size")
        labels = [s.size_label for s in self.sizes]
        if len(set(labels)) != len(labels):
            raise ValueError(f"duplicate size labels in {self.garment_id}: {labels}")
        reference = self.sizes[0].regions
        for size in self.sizes[1:]:
            missing = reference - size.regions
            extra = size.regions - reference
            if missing or extra:
                names = sorted(r.name for r in (missing | extra))
                raise SizeSpecIncomplete(
                    f"size {size.size_label!r} of {self.garment_id} does not measure the same "
                    f"regions as the rest of the run: {', '.join(names)}"
                )
        object.__setattr__(self, "sizes", tuple(self.sizes))

    @property
    def version_key(self) -> str:
        """The identifier recorded in every assessment, so replay can reconstruct inputs."""
        return f"{self.garment_id}@{self.version}"

    @property
    def size_labels(self) -> tuple[str, ...]:
        return tuple(s.size_label for s in self.sizes)

    def size(self, label: str) -> GarmentSizeSpec:
        for candidate in self.sizes:
            if candidate.size_label == label:
                return candidate
        raise SizeNotFound(f"{self.garment_id} has no size {label!r}; has {self.size_labels}")
