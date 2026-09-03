"""Garment specs carry uncertainty too -- grading tolerance is not zero (design 7.1)."""
import pytest

from fitkit.domain.fabric import FabricSpec, RecoveryClass, StretchClass
from fitkit.domain.garment import GarmentSizeSpec, GarmentSpec
from fitkit.domain.errors import SizeNotFound, SizeSpecIncomplete
from fitkit.domain.regions import BodyRegion, GarmentCategory, GarmentRegion, FitIntent

from tests.domain.factories import spec

RIGID = FabricSpec(stretch_class=StretchClass.NONE, recovery=RecoveryClass.GOOD)
JERSEY = FabricSpec(
    stretch_class=StretchClass.HIGH,
    recovery=RecoveryClass.GOOD,
    elongation_pct=35.0,
    elongation_load_n=15.0,
    composition="92% cotton, 8% elastane",
)


def size(label: str, waist: float, hip: float) -> GarmentSizeSpec:
    return GarmentSizeSpec(
        size_label=label,
        measurements={
            GarmentRegion.WAIST_FLAT: spec(waist),
            GarmentRegion.HIP_FLAT: spec(hip),
        },
    )


def trousers(*sizes: GarmentSizeSpec, version: int = 1) -> GarmentSpec:
    return GarmentSpec(
        garment_id="brand:sku-1",
        version=version,
        category=GarmentCategory.TROUSERS,
        size_system="EU",
        fit_intent=FitIntent.REGULAR,
        fabric=RIGID,
        sizes=sizes or (size("46", 39.0, 47.0), size("48", 41.0, 49.0)),
    )


class TestGarmentRegionVocabulary:
    def test_every_garment_region_maps_to_a_body_region(self):
        """The correspondence must be total, or the engine has regions it cannot compare."""
        for region in GarmentRegion:
            assert isinstance(region.body_region, BodyRegion)

    def test_flat_regions_are_marked_as_flat(self):
        assert GarmentRegion.WAIST_FLAT.is_flat is True
        assert GarmentRegion.INSEAM.is_flat is False

    def test_flatness_is_declared_for_every_region(self):
        for region in GarmentRegion:
            assert isinstance(region.is_flat, bool)


class TestSizeSpec:
    def test_measurements_are_immutable(self):
        s = size("48", 41.0, 49.0)
        with pytest.raises(TypeError):
            s.measurements[GarmentRegion.THIGH_FLAT] = spec(30.0)  # type: ignore[index]

    def test_rejects_an_empty_measurement_set(self):
        with pytest.raises(ValueError, match="at least one"):
            GarmentSizeSpec(size_label="48", measurements={})

    def test_rejects_a_blank_size_label(self):
        with pytest.raises(ValueError, match="size_label"):
            GarmentSizeSpec(size_label="  ", measurements={GarmentRegion.WAIST_FLAT: spec(41.0)})

    def test_rejects_non_positive_measurements(self):
        with pytest.raises(ValueError, match="WAIST_FLAT"):
            GarmentSizeSpec(size_label="48", measurements={GarmentRegion.WAIST_FLAT: spec(-1.0)})

    def test_a_spec_measurement_still_cannot_be_certain(self):
        """Zero grading tolerance is the garment-side form of false precision."""
        from fitkit.domain.units import Measure, MeasureSource

        with pytest.raises(ValueError, match="sigma_cm"):
            Measure(41.0, 0.0, MeasureSource.SPEC_SHEET)


class TestGarmentSpec:
    def test_looks_up_a_size_by_label(self):
        g = trousers()
        assert g.size("48").measurements[GarmentRegion.WAIST_FLAT].value_cm == 41.0

    def test_unknown_size_raises_a_typed_error(self):
        with pytest.raises(SizeNotFound):
            trousers().size("52")

    def test_rejects_duplicate_size_labels(self):
        with pytest.raises(ValueError, match="duplicate"):
            trousers(size("48", 41.0, 49.0), size("48", 42.0, 50.0))

    def test_rejects_an_empty_size_run(self):
        with pytest.raises(ValueError, match="at least one size"):
            GarmentSpec(
                garment_id="brand:sku-1",
                version=1,
                category=GarmentCategory.TROUSERS,
                size_system="EU",
                fit_intent=FitIntent.REGULAR,
                fabric=RIGID,
                sizes=(),
            )

    def test_rejects_a_non_positive_version(self):
        with pytest.raises(ValueError, match="version"):
            trousers(version=0)

    def test_rejects_a_ragged_size_run(self):
        """Every size must measure the same regions, or cross-size ranking is incoherent."""
        ragged = GarmentSizeSpec(
            size_label="50", measurements={GarmentRegion.WAIST_FLAT: spec(43.0)}
        )
        with pytest.raises(SizeSpecIncomplete, match="HIP_FLAT"):
            trousers(size("48", 41.0, 49.0), ragged)

    def test_version_key_identifies_the_exact_spec(self):
        assert trousers(version=7).version_key == "brand:sku-1@7"


class TestFabricSpec:
    def test_a_stretch_figure_requires_the_load_it_was_measured_at(self):
        """Open question 11 encoded in the type: 'stretch %' alone is not a quantity."""
        with pytest.raises(ValueError, match="elongation_load_n"):
            FabricSpec(
                stretch_class=StretchClass.HIGH,
                recovery=RecoveryClass.GOOD,
                elongation_pct=35.0,
            )

    def test_a_load_without_a_figure_is_also_rejected(self):
        with pytest.raises(ValueError, match="elongation_pct"):
            FabricSpec(
                stretch_class=StretchClass.HIGH,
                recovery=RecoveryClass.GOOD,
                elongation_load_n=15.0,
            )

    def test_a_class_only_spec_is_valid(self):
        assert RIGID.elongation_pct is None

    def test_a_fully_specified_fabric_is_valid(self):
        assert JERSEY.elongation_pct == 35.0
        assert JERSEY.elongation_load_n == 15.0

    def test_rejects_negative_elongation(self):
        with pytest.raises(ValueError, match="elongation_pct"):
            FabricSpec(
                stretch_class=StretchClass.HIGH,
                recovery=RecoveryClass.GOOD,
                elongation_pct=-1.0,
                elongation_load_n=15.0,
            )
