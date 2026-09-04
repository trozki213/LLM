"""The contract every measurement backend must satisfy.

C5 says the backend will be replaced. This suite is what "replaced successfully" means:
a new backend is done when it passes these, whoever wrote it.
"""

from __future__ import annotations

import pytest

from fitkit.domain.body import BodyMeasurements
from fitkit.domain.units import MIN_SIGMA_CM


class MeasurementBackendContract:
    """Mix in and provide `backend`, `bundle` and `calibration` fixtures."""

    def test_returns_body_measurements(self, backend, bundle, calibration):
        assert isinstance(backend.estimate(bundle, calibration), BodyMeasurements)

    def test_reports_only_regions_it_claims_to_support(self, backend, bundle, calibration):
        result = backend.estimate(bundle, calibration)
        assert result.regions <= backend.supported_regions

    def test_every_measurement_is_positive_centimetres(self, backend, bundle, calibration):
        for region, measure in backend.estimate(bundle, calibration).values.items():
            assert measure.value_cm > 0, region
            assert 20.0 < measure.value_cm < 300.0, region

    def test_every_measurement_carries_uncertainty(self, backend, bundle, calibration):
        for region, measure in backend.estimate(bundle, calibration).values.items():
            assert measure.sigma_cm >= MIN_SIGMA_CM, region

    def test_uncertainty_exceeds_the_shared_scale_component(self, backend, bundle, calibration):
        result = backend.estimate(bundle, calibration)
        for region, measure in result.values.items():
            assert measure.sigma_cm >= calibration.sigma_rel * measure.value_cm - 1e-9, region

    def test_provenance_is_fully_populated(self, backend, bundle, calibration):
        p = backend.estimate(bundle, calibration).provenance
        for field in (
            "backend_id", "backend_version", "residual_table_version",
            "capture_id", "calibration_source_id",
        ):
            assert getattr(p, field), field
        assert p.computed_at.tzinfo is not None

    def test_provenance_names_the_capture_and_the_calibration_used(
        self, backend, bundle, calibration
    ):
        p = backend.estimate(bundle, calibration).provenance
        assert p.capture_id == bundle.capture_id
        assert p.calibration_source_id == calibration.source_id

    def test_the_same_input_gives_the_same_answer(self, backend, bundle, calibration):
        first = backend.estimate(bundle, calibration)
        second = backend.estimate(bundle, calibration)
        assert first.values == second.values
