"""Catalog: importers, the builder gate, and versioned immutable storage."""
import pytest

from fitkit.catalog import (
    CsvSpecImporter,
    GarmentSpecBuilder,
    ImportRow,
    InMemoryGarmentRepository,
    JsonFileGarmentRepository,
)
from fitkit.domain.errors import GarmentNotFound, SizeSpecIncomplete, StorageError
from fitkit.domain.fabric import FabricSpec, RecoveryClass, StretchClass
from fitkit.domain.regions import FitIntent, GarmentCategory, GarmentRegion

RIGID = FabricSpec(StretchClass.NONE, RecoveryClass.GOOD)

GOOD_CSV = b"""size_label,region,value_cm,tolerance_cm
46,waist_flat,39.0,0.5
46,hip_flat,47.0,0.5
48,waist_flat,41.0,
48,hip_flat,49.0,0.6
"""


def builder(**overrides) -> GarmentSpecBuilder:
    b = GarmentSpecBuilder().with_identity(
        overrides.get("garment_id", "brand:sku-1"),
        version=overrides.get("version", 1),
        category=GarmentCategory.TROUSERS,
        size_system="EU",
        fit_intent=FitIntent.REGULAR,
    )
    return b.with_fabric(RIGID).with_grading_tolerance(overrides.get("tolerance", 0.8))


class TestCsvImporter:
    def test_parses_every_row(self):
        result = CsvSpecImporter().parse(GOOD_CSV)
        assert result.ok
        assert len(result.rows) == 4
        assert result.rows[0] == ImportRow("46", GarmentRegion.WAIST_FLAT, 39.0, 0.5)

    def test_a_blank_tolerance_defers_to_the_brand_default(self):
        rows = CsvSpecImporter().parse(GOOD_CSV).rows
        assert rows[2].tolerance_cm is None

    def test_reports_a_bad_row_without_losing_the_good_ones(self):
        raw = GOOD_CSV + b"50,elbow_flat,20.0,0.5\n"
        result = CsvSpecImporter().parse(raw)
        assert len(result.rows) == 4
        assert not result.ok
        assert result.diagnostics[0].line == 6
        assert "elbow_flat" in result.diagnostics[0].detail

    def test_reports_a_non_numeric_measurement(self):
        result = CsvSpecImporter().parse(b"size_label,region,value_cm\n48,waist_flat,about 41\n")
        assert result.diagnostics[0].code == "unparseable_row"

    def test_a_blank_size_label_is_reported(self):
        result = CsvSpecImporter().parse(b"size_label,region,value_cm\n ,waist_flat,41.0\n")
        assert "size_label" in result.diagnostics[0].detail

    def test_missing_columns_are_reported_once(self):
        result = CsvSpecImporter().parse(b"size,region\n46,waist_flat\n")
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].code == "missing_columns"

    def test_tolerates_a_utf8_bom(self):
        assert CsvSpecImporter().parse(b"\xef\xbb\xbf" + GOOD_CSV).ok


class TestBuilderIsTheValidationGate:
    def test_builds_a_spec_from_imported_rows(self):
        rows = CsvSpecImporter().parse(GOOD_CSV).rows
        spec = builder().with_rows(rows).build()
        assert spec.size_labels == ("46", "48")
        assert spec.version_key == "brand:sku-1@1"

    def test_applies_the_brand_tolerance_where_a_row_has_none(self):
        rows = CsvSpecImporter().parse(GOOD_CSV).rows
        spec = builder(tolerance=0.8).with_rows(rows).build()
        assert spec.size("48").measurements[GarmentRegion.WAIST_FLAT].sigma_cm == 0.8
        assert spec.size("48").measurements[GarmentRegion.HIP_FLAT].sigma_cm == 0.6

    def test_a_zero_grading_tolerance_is_refused(self):
        with pytest.raises(ValueError, match="tolerance"):
            builder().with_grading_tolerance(0.0)

    def test_cannot_build_without_a_fabric(self):
        b = GarmentSpecBuilder().with_identity(
            "brand:sku-1", version=1, category=GarmentCategory.TROUSERS, size_system="EU"
        )
        with pytest.raises(SizeSpecIncomplete, match="fabric"):
            b.with_grading_tolerance(0.5).with_rows(CsvSpecImporter().parse(GOOD_CSV).rows).build()

    def test_cannot_build_without_a_tolerance(self):
        b = GarmentSpecBuilder().with_identity(
            "brand:sku-1", version=1, category=GarmentCategory.TROUSERS, size_system="EU"
        ).with_fabric(RIGID)
        with pytest.raises(SizeSpecIncomplete, match="tolerance"):
            b.with_rows(CsvSpecImporter().parse(GOOD_CSV).rows).build()

    def test_cannot_build_with_no_measurements(self):
        with pytest.raises(SizeSpecIncomplete, match="no measurements"):
            builder().build()

    def test_a_duplicated_region_is_refused(self):
        rows = list(CsvSpecImporter().parse(GOOD_CSV).rows)
        rows.append(ImportRow("46", GarmentRegion.WAIST_FLAT, 40.0, 0.5))
        with pytest.raises(SizeSpecIncomplete, match="twice"):
            builder().with_rows(rows).build()

    def test_a_ragged_run_is_refused_by_the_domain(self):
        rows = list(CsvSpecImporter().parse(GOOD_CSV).rows)
        rows.append(ImportRow("50", GarmentRegion.WAIST_FLAT, 43.0, 0.5))
        with pytest.raises(SizeSpecIncomplete, match="HIP_FLAT"):
            builder().with_rows(rows).build()

    def test_numeric_size_labels_sort_numerically(self):
        rows = [
            ImportRow(label, GarmentRegion.WAIST_FLAT, 39.0, 0.5)
            for label in ("50", "46", "48")
        ]
        assert builder().with_rows(rows).build().size_labels == ("46", "48", "50")

    def test_lettered_size_labels_keep_a_stable_order(self):
        rows = [
            ImportRow(label, GarmentRegion.WAIST_FLAT, 39.0, 0.5) for label in ("M", "L", "S")
        ]
        assert builder().with_rows(rows).build().size_labels == ("L", "M", "S")


def _repositories(tmp_path):
    return [InMemoryGarmentRepository(), JsonFileGarmentRepository(tmp_path)]


class TestRepositoryContract:
    """One suite, two implementations. That is the whole point of the port."""

    @pytest.fixture(params=["memory", "json"])
    def repo(self, request, tmp_path):
        return InMemoryGarmentRepository() if request.param == "memory" else JsonFileGarmentRepository(tmp_path)

    def test_round_trips_a_spec(self, repo):
        spec = builder().with_rows(CsvSpecImporter().parse(GOOD_CSV).rows).build()
        repo.add(spec)
        assert repo.get("brand:sku-1", 1) == spec

    def test_defaults_to_the_latest_version(self, repo):
        rows = CsvSpecImporter().parse(GOOD_CSV).rows
        repo.add(builder(version=1).with_rows(rows).build())
        repo.add(builder(version=2).with_rows(rows).build())
        assert repo.get("brand:sku-1").version == 2
        assert repo.latest_version("brand:sku-1") == 2

    def test_an_old_version_stays_retrievable_forever(self, repo):
        rows = CsvSpecImporter().parse(GOOD_CSV).rows
        repo.add(builder(version=1, tolerance=0.8).with_rows(rows).build())
        repo.add(builder(version=2, tolerance=1.4).with_rows(rows).build())
        old = repo.get("brand:sku-1", 1)
        assert old.size("48").measurements[GarmentRegion.WAIST_FLAT].sigma_cm == 0.8

    def test_overwriting_a_published_version_is_refused(self, repo):
        spec = builder().with_rows(CsvSpecImporter().parse(GOOD_CSV).rows).build()
        repo.add(spec)
        with pytest.raises(StorageError, match="already exists"):
            repo.add(spec)

    def test_an_unknown_garment_raises_a_typed_error(self, repo):
        with pytest.raises(GarmentNotFound):
            repo.get("brand:nope")

    def test_an_unknown_version_raises_a_typed_error(self, repo):
        repo.add(builder().with_rows(CsvSpecImporter().parse(GOOD_CSV).rows).build())
        with pytest.raises(GarmentNotFound):
            repo.get("brand:sku-1", 99)

    def test_uncertainty_survives_the_round_trip(self, repo):
        spec = builder(tolerance=0.8).with_rows(CsvSpecImporter().parse(GOOD_CSV).rows).build()
        repo.add(spec)
        restored = repo.get("brand:sku-1", 1)
        assert restored.size("48").measurements[GarmentRegion.WAIST_FLAT].sigma_cm == 0.8
