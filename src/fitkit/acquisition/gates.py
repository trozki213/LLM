"""Capture-quality gates.

Composite earns its place because the gate set grows, differs between the frontal and
lateral views, and a merchant may want one relaxed. Without it the caller hard-codes a
list and gate composition can neither be configured nor tested as a unit.

Chain of Responsibility was considered and rejected: fail-fast ordering with per-gate
remediation is a loop over an ordered list, and four classes to express `break` is the
gratuitous indirection this design treats as a defect.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from fitkit.domain.capture import FrameSignals, GateVerdict, RemediationCode, ViewKind


class CaptureQualityGate(typing.Protocol):
    gate_id: str

    def evaluate(self, signals: FrameSignals) -> GateVerdict: ...


@dataclass(frozen=True, slots=True)
class ThresholdGate:
    """One normalised signal against one floor. Most gates are exactly this."""

    gate_id: str
    attribute: str
    minimum: float
    remediation: RemediationCode

    def evaluate(self, signals: FrameSignals) -> GateVerdict:
        score = getattr(signals, self.attribute)
        passed = score >= self.minimum
        return GateVerdict(
            gate_id=self.gate_id,
            passed=passed,
            score=score,
            remediation=None if passed else self.remediation,
        )


@dataclass(frozen=True, slots=True)
class FramingGate:
    """Head and feet both in frame. Not a threshold, so it is not a ThresholdGate."""

    gate_id: str = "framing"

    def evaluate(self, signals: FrameSignals) -> GateVerdict:
        passed = signals.head_visible and signals.feet_visible
        return GateVerdict(
            gate_id=self.gate_id,
            passed=passed,
            score=1.0 if passed else 0.0,
            remediation=None if passed else RemediationCode.FULL_BODY_IN_FRAME,
        )


@dataclass(frozen=True, slots=True)
class DistanceGate:
    """The subject should fill the frame without touching its edges."""

    gate_id: str = "distance"
    minimum: float = 0.55
    maximum: float = 0.95

    def evaluate(self, signals: FrameSignals) -> GateVerdict:
        fraction = signals.subject_frame_fraction
        if fraction < self.minimum:
            remediation = RemediationCode.STEP_CLOSER
        elif fraction > self.maximum:
            remediation = RemediationCode.STEP_BACK
        else:
            remediation = None
        return GateVerdict(self.gate_id, remediation is None, fraction, remediation)


@dataclass(frozen=True, slots=True)
class DeviceTiltGate:
    gate_id: str = "device_tilt"
    max_pitch_deg: float = 8.0

    def evaluate(self, signals: FrameSignals) -> GateVerdict:
        pitch = abs(signals.device_pitch_deg)
        passed = pitch <= self.max_pitch_deg
        # Reported as a normalised score so every verdict is comparable.
        score = max(0.0, 1.0 - pitch / max(self.max_pitch_deg * 3, 1e-9))
        return GateVerdict(
            self.gate_id, passed, min(score, 1.0),
            None if passed else RemediationCode.STRAIGHTEN_DEVICE,
        )


class CompositeGate:
    """A gate made of gates. Evaluates all of them, so the user gets every problem at once."""

    def __init__(self, gate_id: str, children: tuple[CaptureQualityGate, ...]) -> None:
        if not children:
            raise ValueError("a composite gate must have at least one child")
        self.gate_id = gate_id
        self._children = children

    def evaluate(self, signals: FrameSignals) -> GateVerdict:
        verdicts = self.evaluate_all(signals)
        failed = [v for v in verdicts if not v.passed]
        if not failed:
            return GateVerdict(self.gate_id, True, min(v.score for v in verdicts), None)
        return GateVerdict(
            self.gate_id, False, min(v.score for v in failed), failed[0].remediation
        )

    def evaluate_all(self, signals: FrameSignals) -> tuple[GateVerdict, ...]:
        return tuple(child.evaluate(signals) for child in self._children)


def standard_gates(view: ViewKind) -> CompositeGate:
    """The launch protocol. Ordered so the most actionable problem is reported first.

    Thresholds are provisional. Phase 7 relates gate scores to measurement error, and
    that is what should set them -- a gate that rejects good captures is a conversion
    problem, and one that accepts bad ones is a silent accuracy problem.
    """
    common: tuple[CaptureQualityGate, ...] = (
        FramingGate(),
        DistanceGate(),
        ThresholdGate("sharpness", "sharpness", 0.6, RemediationCode.HOLD_STILL),
        ThresholdGate("exposure", "exposure", 0.45, RemediationCode.MORE_LIGHT),
        ThresholdGate(
            "background", "background_separability", 0.5, RemediationCode.PLAIN_BACKGROUND
        ),
        DeviceTiltGate(),
        ThresholdGate(
            "clothing", "clothing_tightness", 0.5, RemediationCode.TIGHTER_CLOTHING
        ),
        ThresholdGate("posture", "torso_verticality", 0.7, RemediationCode.STRAIGHTEN_DEVICE),
    )
    if view is ViewKind.FRONTAL:
        common += (
            ThresholdGate(
                "arm_separation", "arm_separation", 0.5, RemediationCode.ARMS_AWAY_FROM_BODY
            ),
        )
    return CompositeGate(f"standard/{view.value}", common)
