"""Turning raw frames into a validated CaptureBundle, or a rejection you can act on."""

from __future__ import annotations

from dataclasses import dataclass

from fitkit.domain.capture import (
    CaptureBundle,
    DeviceMetadata,
    GateVerdict,
    ViewKind,
)
from fitkit.domain.errors import CaptureRejected
from fitkit.domain.ports import FrameAnalyzer, PhotoStore
from fitkit.domain.units import Mass, Measure, MeasureSource
from fitkit.acquisition.gates import CompositeGate, standard_gates

#: Standard deviation of self-reported height, in centimetres. People round to the
#: nearest five and quote figures from decades ago. This is a placeholder: design 7.4
#: says the value must be measured on the validation panel (declared vs tape), and
#: Phase 7 produces it. It is deliberately not zero, because zero is the exact false
#: precision C6 exists to prevent, entering through the calibration door.
DEFAULT_HEIGHT_SIGMA_CM = 1.5
DEFAULT_WEIGHT_SIGMA_KG = 2.0


@dataclass(frozen=True, slots=True)
class RawCapture:
    frontal: bytes
    lateral: bytes
    declared_height_cm: float
    declared_weight_kg: float | None
    device: DeviceMetadata


class CaptureAssembler:
    def __init__(
        self,
        analyzer: FrameAnalyzer,
        photos: PhotoStore,
        *,
        frontal_gate: CompositeGate | None = None,
        lateral_gate: CompositeGate | None = None,
        height_sigma_cm: float = DEFAULT_HEIGHT_SIGMA_CM,
        weight_sigma_kg: float = DEFAULT_WEIGHT_SIGMA_KG,
    ) -> None:
        self._analyzer = analyzer
        self._photos = photos
        self._frontal = frontal_gate or standard_gates(ViewKind.FRONTAL)
        self._lateral = lateral_gate or standard_gates(ViewKind.LATERAL)
        self._height_sigma = height_sigma_cm
        self._weight_sigma = weight_sigma_kg

    def assemble(self, capture_id: str, raw: RawCapture) -> CaptureBundle:
        verdicts: list[GateVerdict] = []
        for image, view, gate in (
            (raw.frontal, ViewKind.FRONTAL, self._frontal),
            (raw.lateral, ViewKind.LATERAL, self._lateral),
        ):
            signals = self._analyzer.analyze(image, view)
            verdicts.extend(
                GateVerdict(
                    gate_id=f"{view.value}.{v.gate_id}",
                    passed=v.passed,
                    score=v.score,
                    remediation=v.remediation,
                )
                for v in gate.evaluate_all(signals)
            )

        failed = tuple(v.gate_id for v in verdicts if not v.passed)
        if failed:
            raise CaptureRejected(failed)

        frontal_ref = self._photos.put(capture_id, ViewKind.FRONTAL.value, raw.frontal)
        lateral_ref = self._photos.put(capture_id, ViewKind.LATERAL.value, raw.lateral)
        return CaptureBundle(
            capture_id=capture_id,
            frontal=frontal_ref,
            lateral=lateral_ref,
            declared_height=Measure(
                raw.declared_height_cm, self._height_sigma, MeasureSource.USER_DECLARED
            ),
            declared_weight=(
                None
                if raw.declared_weight_kg is None
                else Mass(raw.declared_weight_kg, self._weight_sigma, MeasureSource.USER_DECLARED)
            ),
            device=raw.device,
            gate_report=tuple(verdicts),
        )

    @staticmethod
    def remediations(error: CaptureRejected, verdicts: tuple[GateVerdict, ...]) -> tuple[str, ...]:
        return tuple(
            v.remediation.value for v in verdicts if not v.passed and v.remediation is not None
        )
