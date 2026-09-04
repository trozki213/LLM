"""Measurement: calibration, measured residuals, and the vendor adapter."""
import datetime as dt

import pytest

from fitkit.domain.capture import CaptureBundle, DeviceMetadata, GateVerdict, PhotoRef
from fitkit.domain.errors import (
    BackendTimeout,
    BackendUnavailable,
    InvalidDeclaredHeight,
    UncalibratedBackend,
)
from fitkit.domain.regions import BodyRegion
from fitkit.domain.units import Mass, Measure, MeasureSource
from fitkit.measurement import (
    DeclaredHeightCalibration,
    ResidualEntry,
    ResidualTable,
    UncertaintyCalibrator,
    VendorMeasurementBackend,
)

from tests.fakes import FixedMeasurementBackend
from tests.measurement.contract import MeasurementBackendContract


def bundle(height_cm=175.0, height_sigma=1.5, weight=None) -> CaptureBundle:
    return CaptureBundle(
        capture_id="cap_01J",
        frontal=PhotoRef("s3://c/f.jpg", "a" * 64),
        lateral=PhotoRef("s3://c/l.jpg", "b" * 64),
        declared_height=Measure(height_cm, height_sigma, MeasureSource.USER_DECLARED),
        declared_weight=None if weight is None else Mass(weight, 2.0, MeasureSource.USER_DECLARED),
        device=DeviceMetadata("ios", "iPhone15,2", "1.0.0"),
        gate_report=(GateVerdict("framing", True, 1.0, None),),
    )


TABLE = ResidualTable(
    version="residuals/2026-09",
    entries=(
        ResidualEntry("fixed", BodyRegion.WAIST, 85.0, 1.6),
        ResidualEntry("fixed", BodyRegion.WAIST, 999.0, 2.4),
        ResidualEntry("fixed", BodyRegion.HIP, 999.0, 1.3),
        ResidualEntry("vendor", BodyRegion.WAIST, 999.0, 1.9),
        ResidualEntry("vendor", BodyRegion.HIP, 999.0, 1.5),
        ResidualEntry("vendor", BodyRegion.BUST, 999.0, 2.1),
        ResidualEntry("vendor", BodyRegion.THIGH, 999.0, 1.4),
        ResidualEntry("vendor", BodyRegion.INSEAM, 999.0, 1.1),
    ),
)


class StubTransport:
    def __init__(self, response=None, error=None):
        self._response = response or {
            "measurements": {"waist": 80.4, "hip": 95.1, "bust": 90.0, "elbow": 1.0},
            "computed_at": "2026-09-03T11:04:22+00:00",
        }
        self._error = error
        self.calls = []

    def post_json(self, url, payload, *, timeout_s):
        self.calls.append((url, payload, timeout_s))
        if self._error:
            raise self._error
        return self._response


class TestDeclaredHeightCalibration:
    def test_relative_sigma_follows_the_declared_uncertainty(self):
        cal = DeclaredHeightCalibration().calibrate(bundle(175.0, 1.5))
        assert cal.sigma_rel == pytest.approx(1.5 / 175.0)

    def test_carries_the_reference_it_used(self):
        assert DeclaredHeightCalibration().calibrate(bundle()).reference.value_cm == 175.0

    def test_a_more_certain_height_gives_a_tighter_scale(self):
        loose = DeclaredHeightCalibration().calibrate(bundle(175.0, 3.0)).sigma_rel
        tight = DeclaredHeightCalibration().calibrate(bundle(175.0, 0.5)).sigma_rel
        assert tight < loose

    def test_it_is_one_implementation_of_a_seam_not_the_seam_itself(self):
        """C3: swapping in a depth source must not touch anything downstream."""
        from fitkit.domain.ports import ScaleCalibrationSource

        class DepthCalibration:
            source_id = "arkit-depth/1"

            def calibrate(self, bundle):
                from fitkit.domain.body import ScaleCalibration

                return ScaleCalibration(source_id=self.source_id, sigma_rel=0.002)

        source: ScaleCalibrationSource = DepthCalibration()
        assert source.calibrate(bundle()).sigma_rel < DeclaredHeightCalibration().calibrate(
            bundle()
        ).sigma_rel


class TestResidualTable:
    def test_selects_the_bucket_the_measurement_falls_into(self):
        assert TABLE.residual_cm("fixed", BodyRegion.WAIST, 80.0) == 1.6
        assert TABLE.residual_cm("fixed", BodyRegion.WAIST, 95.0) == 2.4

    def test_an_uncharacterised_backend_fails_closed(self):
        with pytest.raises(UncalibratedBackend, match="validation panel"):
            TABLE.residual_cm("brand-new", BodyRegion.WAIST, 80.0)

    def test_an_uncharacterised_region_fails_closed(self):
        with pytest.raises(UncalibratedBackend):
            TABLE.residual_cm("fixed", BodyRegion.NECK, 38.0)

    def test_a_zero_residual_cannot_be_recorded(self):
        with pytest.raises(ValueError, match="residual_cm"):
            ResidualEntry("fixed", BodyRegion.WAIST, 999.0, 0.0)

    def test_a_table_must_be_versioned(self):
        with pytest.raises(ValueError, match="versioned"):
            ResidualTable(version="  ", entries=())

    def test_entries_are_ordered_deterministically(self):
        shuffled = ResidualTable(version="v", entries=tuple(reversed(TABLE.entries)))
        assert shuffled.entries == TABLE.entries


class TestUncertaintyCalibrator:
    def _calibrated(self):
        return UncertaintyCalibrator(FixedMeasurementBackend(waist_cm=80.0), TABLE)

    def test_replaces_the_backends_own_sigma_with_the_measured_one(self):
        cal = DeclaredHeightCalibration().calibrate(bundle())
        raw = FixedMeasurementBackend(waist_cm=80.0).estimate(bundle(), cal)
        fixed = self._calibrated().estimate(bundle(), cal)
        assert raw.residual_sigma_cm(BodyRegion.WAIST) == 1.2
        assert fixed.residual_sigma_cm(BodyRegion.WAIST) == 1.6

    def test_stamps_the_table_version_into_provenance(self):
        cal = DeclaredHeightCalibration().calibrate(bundle())
        result = self._calibrated().estimate(bundle(), cal)
        assert result.provenance.residual_table_version == "residuals/2026-09"

    def test_leaves_the_measured_values_alone(self):
        cal = DeclaredHeightCalibration().calibrate(bundle())
        assert self._calibrated().estimate(bundle(), cal).values[BodyRegion.WAIST].value_cm == 80.0

    def test_a_larger_body_gets_its_own_measured_residual(self):
        cal = DeclaredHeightCalibration().calibrate(bundle())
        big = UncertaintyCalibrator(FixedMeasurementBackend(waist_cm=95.0), TABLE)
        assert big.estimate(bundle(), cal).residual_sigma_cm(BodyRegion.WAIST) == 2.4

    def test_it_presents_the_same_surface_as_the_backend_it_wraps(self):
        inner = FixedMeasurementBackend()
        wrapped = UncertaintyCalibrator(inner, TABLE)
        assert wrapped.backend_id == inner.backend_id
        assert wrapped.supported_regions == inner.supported_regions


class TestVendorAdapter:
    def _backend(self, transport=None):
        return VendorMeasurementBackend(
            transport or StubTransport(), url="https://vendor.example/scan", backend_version="2026.07"
        )

    def test_sends_the_capture_and_the_declared_inputs(self):
        transport = StubTransport()
        cal = DeclaredHeightCalibration().calibrate(bundle(weight=72.0))
        self._backend(transport).estimate(bundle(weight=72.0), cal)
        _, payload, _ = transport.calls[0]
        assert payload["height_cm"] == 175.0
        assert payload["weight_kg"] == 72.0
        assert payload["frontal_uri"].endswith("f.jpg")

    def test_ignores_regions_the_system_does_not_model(self):
        cal = DeclaredHeightCalibration().calibrate(bundle())
        result = self._backend().estimate(bundle(), cal)
        assert BodyRegion.WAIST in result.regions
        assert len(result.regions) == 3

    def test_a_timeout_becomes_a_typed_timeout(self):
        cal = DeclaredHeightCalibration().calibrate(bundle())
        with pytest.raises(BackendTimeout):
            self._backend(StubTransport(error=TimeoutError("slow"))).estimate(bundle(), cal)

    def test_any_other_transport_failure_becomes_backend_unavailable(self):
        cal = DeclaredHeightCalibration().calibrate(bundle())
        with pytest.raises(BackendUnavailable):
            self._backend(StubTransport(error=ConnectionResetError("boom"))).estimate(bundle(), cal)

    def test_no_foreign_exception_type_escapes_the_adapter(self):
        from fitkit.domain.errors import FitKitError

        cal = DeclaredHeightCalibration().calibrate(bundle())
        for error in (TimeoutError(), ConnectionResetError(), ValueError("odd")):
            with pytest.raises(FitKitError):
                self._backend(StubTransport(error=error)).estimate(bundle(), cal)

    def test_an_empty_response_is_refused_rather_than_returned(self):
        cal = DeclaredHeightCalibration().calibrate(bundle())
        with pytest.raises(BackendUnavailable, match="no measurements"):
            self._backend(StubTransport({"measurements": {}})).estimate(bundle(), cal)

    def test_a_naive_timestamp_is_refused(self):
        cal = DeclaredHeightCalibration().calibrate(bundle())
        bad = StubTransport({"measurements": {"waist": 80.0}, "computed_at": "2026-09-03T11:00:00"})
        with pytest.raises(BackendUnavailable, match="timezone"):
            self._backend(bad).estimate(bundle(), cal)

    def test_it_ships_uncalibrated_and_says_so(self):
        cal = DeclaredHeightCalibration().calibrate(bundle())
        result = self._backend().estimate(bundle(), cal)
        assert result.provenance.residual_table_version == "uncalibrated"


class TestFixedBackendMeetsTheContract(MeasurementBackendContract):
    @pytest.fixture
    def backend(self):
        return FixedMeasurementBackend()

    @pytest.fixture
    def bundle(self):
        return globals()["bundle"]()

    @pytest.fixture
    def calibration(self, bundle):
        return DeclaredHeightCalibration().calibrate(bundle)


class TestVendorBackendMeetsTheContract(MeasurementBackendContract):
    @pytest.fixture
    def backend(self):
        return VendorMeasurementBackend(
            StubTransport(), url="https://vendor.example/scan", backend_version="2026.07"
        )

    @pytest.fixture
    def bundle(self):
        return globals()["bundle"]()

    @pytest.fixture
    def calibration(self, bundle):
        return DeclaredHeightCalibration().calibrate(bundle)


class TestCalibratedVendorMeetsTheContract(MeasurementBackendContract):
    """The decorator must not break the contract of the thing it decorates."""

    @pytest.fixture
    def backend(self):
        return UncertaintyCalibrator(
            VendorMeasurementBackend(
                StubTransport(), url="https://vendor.example/scan", backend_version="2026.07"
            ),
            TABLE,
        )

    @pytest.fixture
    def bundle(self):
        return globals()["bundle"]()

    @pytest.fixture
    def calibration(self, bundle):
        return DeclaredHeightCalibration().calibrate(bundle)


class TestCalibrationGuards:
    def test_a_non_positive_declared_height_is_refused(self):
        from fitkit.domain.body import ScaleCalibration

        class Impossible:
            declared_height = type("H", (), {"value_cm": -1.0, "sigma_cm": 1.0})()

        with pytest.raises(InvalidDeclaredHeight):
            DeclaredHeightCalibration().calibrate(Impossible())
