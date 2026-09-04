"""The composition root: the one module allowed to name concrete adapters."""

from __future__ import annotations

import datetime as dt
from typing import Mapping

from fitkit.domain.policy import FitPolicy
from fitkit.domain.ports import AssessmentStore, HttpTransport, MeasurementBackend
from fitkit.explanation.renderers import GuardedRenderer, LlmRenderer
from fitkit.explanation.template import TemplateRenderer
from fitkit.fit_engine.engine import DeterministicFitEngine
from fitkit.measurement.calibration import DeclaredHeightCalibration
from fitkit.measurement.residuals import ResidualTable, UncertaintyCalibrator
from fitkit.measurement.vendor import VendorMeasurementBackend
from fitkit.orchestration.service import SizeAdvisor


class SystemClock:
    def now(self) -> dt.datetime:
        return dt.datetime.now(dt.UTC)


def build_advisor(
    *,
    garments,
    store: AssessmentStore,
    residuals: ResidualTable,
    default_policy: FitPolicy,
    policies: Mapping[str, FitPolicy] | None = None,
    backend: MeasurementBackend | None = None,
    transport: HttpTransport | None = None,
    vendor_url: str = "",
    vendor_version: str = "",
    llm_client=None,
    clock=None,
    metrics=None,
) -> SizeAdvisor:
    """Wire the system.

    The LLM is opt-in: with no client the renderer is the template, which is complete on
    its own (ADR-008, design 7.3). That is C2 expressed as a default rather than as a
    fallback path nobody exercises.
    """
    if backend is None:
        if transport is None:
            raise ValueError("provide either a measurement backend or an HTTP transport")
        backend = VendorMeasurementBackend(
            transport, url=vendor_url, backend_version=vendor_version
        )
    calibrated = UncertaintyCalibrator(backend, residuals)

    template = TemplateRenderer()
    renderer = template if llm_client is None else GuardedRenderer(
        LlmRenderer(llm_client), fallback=template
    )

    return SizeAdvisor(
        calibration=DeclaredHeightCalibration(),
        backend=calibrated,
        garments=garments,
        engine=DeterministicFitEngine(),
        renderer=renderer,
        store=store,
        clock=clock or SystemClock(),
        policies=policies or {},
        default_policy=default_policy,
        metrics=metrics,
    )
