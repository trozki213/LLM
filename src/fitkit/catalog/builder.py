"""Assembling a garment spec from documents that arrive separately.

Builder earns its place because a spec is not one document: the measurement sheet, the
fabric sheet and the grading tolerance turn up at different times and often from
different people. `build()` is the single gate where a half-assembled spec becomes a
`GarmentSpec` -- so a partial one can never leak into the system.
"""

from __future__ import annotations

from typing import Iterable, Self

from fitkit.domain.errors import SizeSpecIncomplete
from fitkit.domain.fabric import FabricSpec
from fitkit.domain.garment import GarmentSizeSpec, GarmentSpec
from fitkit.domain.regions import FitIntent, GarmentCategory, GarmentRegion
from fitkit.domain.units import Measure, MeasureSource
from fitkit.catalog.importers import ImportRow


class GarmentSpecBuilder:
    def __init__(self) -> None:
        self._garment_id: str | None = None
        self._version: int = 1
        self._category: GarmentCategory | None = None
        self._size_system: str = ""
        self._fit_intent: FitIntent = FitIntent.REGULAR
        self._fabric: FabricSpec | None = None
        self._default_tolerance_cm: float | None = None
        self._rows: list[ImportRow] = []

    def with_identity(
        self,
        garment_id: str,
        *,
        version: int,
        category: GarmentCategory,
        size_system: str,
        fit_intent: FitIntent = FitIntent.REGULAR,
    ) -> Self:
        self._garment_id = garment_id
        self._version = version
        self._category = category
        self._size_system = size_system
        self._fit_intent = fit_intent
        return self

    def with_fabric(self, fabric: FabricSpec) -> Self:
        self._fabric = fabric
        return self

    def with_grading_tolerance(self, tolerance_cm: float) -> Self:
        """The brand's stated tolerance, used for any row that does not carry its own."""
        if tolerance_cm <= 0:
            raise ValueError(
                f"grading tolerance must be > 0, got {tolerance_cm!r}; a garment measured "
                "without tolerance is the garment-side form of false precision (design 7.1)"
            )
        self._default_tolerance_cm = tolerance_cm
        return self

    def with_rows(self, rows: Iterable[ImportRow]) -> Self:
        self._rows.extend(rows)
        return self

    def build(self) -> GarmentSpec:
        missing = [
            name
            for name, value in (
                ("identity", self._garment_id),
                ("category", self._category),
                ("fabric", self._fabric),
                ("grading tolerance", self._default_tolerance_cm),
            )
            if value is None
        ]
        if missing:
            raise SizeSpecIncomplete(f"cannot build a garment spec without: {', '.join(missing)}")
        if not self._rows:
            raise SizeSpecIncomplete("cannot build a garment spec with no measurements")

        by_size: dict[str, dict[GarmentRegion, Measure]] = {}
        for row in self._rows:
            tolerance = row.tolerance_cm or self._default_tolerance_cm
            assert tolerance is not None  # guarded above
            bucket = by_size.setdefault(row.size_label, {})
            if row.region in bucket:
                raise SizeSpecIncomplete(
                    f"size {row.size_label!r} measures {row.region.name} twice"
                )
            bucket[row.region] = Measure(row.value_cm, tolerance, MeasureSource.SPEC_SHEET)

        sizes = tuple(
            GarmentSizeSpec(size_label=label, measurements=measurements)
            for label, measurements in sorted(by_size.items(), key=_size_sort_key)
        )
        return GarmentSpec(
            garment_id=self._garment_id,
            version=self._version,
            category=self._category,
            size_system=self._size_system,
            fit_intent=self._fit_intent,
            fabric=self._fabric,
            sizes=sizes,
        )


def _size_sort_key(item: tuple[str, object]) -> tuple[int, float, str]:
    """Numeric labels sort numerically; the rest keep a stable alphabetical order."""
    label = item[0]
    try:
        return (0, float(label), "")
    except ValueError:
        return (1, 0.0, label)
