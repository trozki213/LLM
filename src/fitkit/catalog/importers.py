"""Adapters from whatever a supplier sends to rows this system understands.

Adapter earns its place here because every brand sends a different document. Without
one, format-specific parsing bleeds into the domain and adding a brand becomes a change
to the core.
"""

from __future__ import annotations

import csv
import io
import typing
from dataclasses import dataclass

from fitkit.domain.regions import GarmentRegion


@dataclass(frozen=True, slots=True)
class ImportRow:
    size_label: str
    region: GarmentRegion
    value_cm: float
    tolerance_cm: float | None


@dataclass(frozen=True, slots=True)
class ImportDiagnostic:
    line: int
    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class ImportResult:
    """Rows that parsed, and an account of everything that did not.

    Bad rows are reported rather than raised: a spreadsheet with one broken line should
    tell you about that line, not refuse the other four hundred.
    """

    rows: tuple[ImportRow, ...]
    diagnostics: tuple[ImportDiagnostic, ...]

    @property
    def ok(self) -> bool:
        return not self.diagnostics


class SpecImporter(typing.Protocol):
    source_format: str

    def parse(self, raw: bytes) -> ImportResult: ...


class CsvSpecImporter:
    """Long-format CSV: one row per size and region.

    `size_label,region,value_cm[,tolerance_cm]` -- long rather than wide because a wide
    sheet needs a new column for every region a brand starts measuring, and long does not.
    """

    source_format = "csv/long/1"
    _REQUIRED = ("size_label", "region", "value_cm")

    def parse(self, raw: bytes) -> ImportResult:
        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        rows: list[ImportRow] = []
        problems: list[ImportDiagnostic] = []

        missing = [c for c in self._REQUIRED if c not in (reader.fieldnames or ())]
        if missing:
            return ImportResult(
                (), (ImportDiagnostic(1, "missing_columns", ", ".join(missing)),)
            )

        for line, record in enumerate(reader, start=2):
            try:
                rows.append(self._row(record))
            except (KeyError, ValueError, AttributeError) as exc:
                problems.append(ImportDiagnostic(line, "unparseable_row", str(exc)))
        return ImportResult(tuple(rows), tuple(problems))

    @staticmethod
    def _row(record: dict[str, str]) -> ImportRow:
        label = (record["size_label"] or "").strip()
        if not label:
            raise ValueError("size_label is blank")
        raw_region = (record["region"] or "").strip().lower()
        try:
            region = GarmentRegion(raw_region)
        except ValueError:
            raise ValueError(f"{raw_region!r} is not a known garment region") from None
        tolerance = (record.get("tolerance_cm") or "").strip()
        return ImportRow(
            size_label=label,
            region=region,
            value_cm=float(record["value_cm"]),
            tolerance_cm=float(tolerance) if tolerance else None,
        )
