"""Storage behind the `GarmentRepository` port.

ADR-009: specs are immutable and versioned. An update creates version n+1 and version n
stays retrievable forever, because Phase 7 can only attribute a return to a
recommendation if it can reconstruct the spec that was in force at the time.
"""

from __future__ import annotations

import json
import pathlib

from fitkit.domain.errors import GarmentNotFound, StorageError
from fitkit.domain.fabric import FabricSpec, RecoveryClass, StretchClass
from fitkit.domain.garment import GarmentSizeSpec, GarmentSpec
from fitkit.domain.regions import FitIntent, GarmentCategory, GarmentRegion
from fitkit.domain.units import Measure, MeasureSource


class InMemoryGarmentRepository:
    def __init__(self, *specs: GarmentSpec) -> None:
        self._by_key: dict[tuple[str, int], GarmentSpec] = {}
        for spec in specs:
            self.add(spec)

    def add(self, spec: GarmentSpec) -> None:
        key = (spec.garment_id, spec.version)
        if key in self._by_key:
            raise StorageError(
                f"{spec.version_key} already exists; publish a new version instead of "
                "overwriting one that recommendations may already refer to"
            )
        self._by_key[key] = spec

    def get(self, garment_id: str, version: int | None = None) -> GarmentSpec:
        if version is None:
            version = self.latest_version(garment_id)
        try:
            return self._by_key[(garment_id, version)]
        except KeyError:
            raise GarmentNotFound(f"{garment_id}@{version}") from None

    def latest_version(self, garment_id: str) -> int:
        versions = [v for gid, v in self._by_key if gid == garment_id]
        if not versions:
            raise GarmentNotFound(garment_id)
        return max(versions)


class JsonFileGarmentRepository:
    """One JSON file per version, named `<garment>@<version>.json`. The filesystem is a
    boundary, so it sits behind the same port as everything else."""

    def __init__(self, root: pathlib.Path) -> None:
        self._root = root

    def add(self, spec: GarmentSpec) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._path(spec.garment_id, spec.version)
        if path.exists():
            raise StorageError(f"{spec.version_key} already exists")
        path.write_text(json.dumps(_to_dict(spec), sort_keys=True, indent=2))

    def get(self, garment_id: str, version: int | None = None) -> GarmentSpec:
        if version is None:
            version = self.latest_version(garment_id)
        path = self._path(garment_id, version)
        if not path.exists():
            raise GarmentNotFound(f"{garment_id}@{version}")
        return _from_dict(json.loads(path.read_text()))

    def latest_version(self, garment_id: str) -> int:
        versions = [
            int(p.stem.rsplit("@", 1)[1])
            for p in self._root.glob(f"{_slug(garment_id)}@*.json")
        ]
        if not versions:
            raise GarmentNotFound(garment_id)
        return max(versions)

    def _path(self, garment_id: str, version: int) -> pathlib.Path:
        return self._root / f"{_slug(garment_id)}@{version}.json"


def _slug(garment_id: str) -> str:
    return garment_id.replace("/", "_").replace(":", "_")


def _to_dict(spec: GarmentSpec) -> dict:
    return {
        "garment_id": spec.garment_id,
        "version": spec.version,
        "category": spec.category.value,
        "size_system": spec.size_system,
        "fit_intent": spec.fit_intent.value,
        "fabric": {
            "stretch_class": spec.fabric.stretch_class.value,
            "recovery": spec.fabric.recovery.value,
            "elongation_pct": spec.fabric.elongation_pct,
            "elongation_load_n": spec.fabric.elongation_load_n,
            "composition": spec.fabric.composition,
        },
        "sizes": [
            {
                "size_label": s.size_label,
                "measurements": {
                    region.value: {"value_cm": m.value_cm, "sigma_cm": m.sigma_cm}
                    for region, m in sorted(s.measurements.items(), key=lambda kv: kv[0].value)
                },
            }
            for s in spec.sizes
        ],
    }


def _from_dict(doc: dict) -> GarmentSpec:
    fabric = doc["fabric"]
    return GarmentSpec(
        garment_id=doc["garment_id"],
        version=doc["version"],
        category=GarmentCategory(doc["category"]),
        size_system=doc["size_system"],
        fit_intent=FitIntent(doc["fit_intent"]),
        fabric=FabricSpec(
            stretch_class=StretchClass(fabric["stretch_class"]),
            recovery=RecoveryClass(fabric["recovery"]),
            elongation_pct=fabric.get("elongation_pct"),
            elongation_load_n=fabric.get("elongation_load_n"),
            composition=fabric.get("composition"),
        ),
        sizes=tuple(
            GarmentSizeSpec(
                size_label=s["size_label"],
                measurements={
                    GarmentRegion(region): Measure(
                        m["value_cm"], m["sigma_cm"], MeasureSource.SPEC_SHEET
                    )
                    for region, m in s["measurements"].items()
                },
            )
            for s in doc["sizes"]
        ),
    )
