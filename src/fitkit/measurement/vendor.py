"""A commercial measurement vendor behind the port (ADR-002).

The adapter talks to an injected `HttpTransport` rather than a networking library, so it
is fully unit-testable and adds no dependency. Vendor error shapes are translated at this
boundary; no foreign exception type reaches the domain.
"""

from __future__ import annotations

import datetime as dt

from fitkit.domain.body import BodyMeasurements, MeasurementProvenance, ScaleCalibration
from fitkit.domain.capture import CaptureBundle
from fitkit.domain.errors import BackendTimeout, BackendUnavailable
from fitkit.domain.ports import HttpTransport
from fitkit.domain.regions import BodyRegion
from fitkit.domain.units import Measure, MeasureSource

#: Placeholder sigma. It is never the sigma that ships: `UncertaintyCalibrator` replaces
#: it with a measured residual, and a backend with no residual table fails closed.
_PROVISIONAL_SIGMA_CM = 2.5


class VendorMeasurementBackend:
    backend_id = "vendor"

    def __init__(
        self,
        transport: HttpTransport,
        *,
        url: str,
        backend_version: str,
        timeout_s: float = 10.0,
        regions: frozenset[BodyRegion] | None = None,
    ) -> None:
        self._transport = transport
        self._url = url
        self._version = backend_version
        self._timeout = timeout_s
        self._regions = regions or frozenset(
            {BodyRegion.BUST, BodyRegion.WAIST, BodyRegion.HIP, BodyRegion.THIGH, BodyRegion.INSEAM}
        )

    @property
    def supported_regions(self) -> frozenset[BodyRegion]:
        return self._regions

    def estimate(self, bundle: CaptureBundle, calibration: ScaleCalibration) -> BodyMeasurements:
        payload = {
            "capture_id": bundle.capture_id,
            "frontal_uri": bundle.frontal.uri,
            "lateral_uri": bundle.lateral.uri,
            "height_cm": bundle.declared_height.value_cm,
            "weight_kg": (
                None if bundle.declared_weight is None else bundle.declared_weight.value_kg
            ),
        }
        try:
            response = self._transport.post_json(self._url, payload, timeout_s=self._timeout)
        except TimeoutError as exc:
            raise BackendTimeout(f"{self.backend_id} timed out after {self._timeout}s") from exc
        except Exception as exc:
            raise BackendUnavailable(f"{self.backend_id}: {exc}") from exc

        raw = response.get("measurements")
        if not isinstance(raw, dict) or not raw:
            raise BackendUnavailable(f"{self.backend_id} returned no measurements")

        residuals: dict[BodyRegion, Measure] = {}
        for name, value in sorted(raw.items()):
            try:
                region = BodyRegion(name)
            except ValueError:
                continue  # a region we do not model; ignoring it is not a silent loss
            if region not in self._regions:
                continue
            residuals[region] = Measure(
                float(value), _PROVISIONAL_SIGMA_CM, MeasureSource.ESTIMATED
            )
        if not residuals:
            raise BackendUnavailable(
                f"{self.backend_id} returned no regions this backend claims to support"
            )

        return BodyMeasurements(
            residuals=residuals,
            scale_sigma_rel=calibration.sigma_rel,
            provenance=MeasurementProvenance(
                backend_id=self.backend_id,
                backend_version=self._version,
                residual_table_version="uncalibrated",
                capture_id=bundle.capture_id,
                calibration_source_id=calibration.source_id,
                computed_at=_parse_time(response.get("computed_at")),
            ),
        )


def _parse_time(raw: object) -> dt.datetime:
    if isinstance(raw, str):
        parsed = dt.datetime.fromisoformat(raw)
        if parsed.tzinfo is not None:
            return parsed
    raise BackendUnavailable("vendor response carried no timezone-aware timestamp")
