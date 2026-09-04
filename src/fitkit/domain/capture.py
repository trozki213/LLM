"""The validated output of acquisition.

A `CaptureBundle` is what acquisition produces once every gate has passed. Its
constructor refuses to build one from a failed gate report, so "a capture that was
rejected" and "a capture we measured" are not the same type of thing and cannot be
confused downstream.

There is deliberately no depth field yet. C3 is served by the `ScaleCalibrationSource`
port, not by a speculative payload: when ARKit/ARCore calibration arrives it adds a
field here that only acquisition (producer) and the new calibrator (consumer) touch,
because everything downstream consumes `ScaleCalibration`, never the raw sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from fitkit.domain.units import Mass, Measure, MeasureSource

_MIN_PLAUSIBLE_HEIGHT_CM = 90.0
_MAX_PLAUSIBLE_HEIGHT_CM = 250.0


class ViewKind(StrEnum):
    FRONTAL = "frontal"
    LATERAL = "lateral"


class RemediationCode(StrEnum):
    """What the user must change. Every rejection maps to one of these."""

    STEP_BACK = "step_back"
    STEP_CLOSER = "step_closer"
    FULL_BODY_IN_FRAME = "full_body_in_frame"
    STRAIGHTEN_DEVICE = "straighten_device"
    MORE_LIGHT = "more_light"
    PLAIN_BACKGROUND = "plain_background"
    ARMS_AWAY_FROM_BODY = "arms_away_from_body"
    HOLD_STILL = "hold_still"
    TIGHTER_CLOTHING = "tighter_clothing"


@dataclass(frozen=True, slots=True)
class PhotoRef:
    """A pointer to an image. Pixels never enter the domain, or the logs."""

    uri: str
    sha256: str

    def __post_init__(self) -> None:
        if len(self.sha256) != 64:
            raise ValueError(f"sha256 must be 64 hex characters, got {len(self.sha256)}")


@dataclass(frozen=True, slots=True)
class DeviceMetadata:
    platform: str
    model: str
    app_version: str


@dataclass(frozen=True, slots=True)
class FrameSignals:
    """What a frame analyser extracted from one photograph.

    Pixels never reach a gate. Gates reason about normalised signals, which is what makes
    them pure functions that can be unit-tested without a single image on disk, and what
    lets the analyser itself be swapped for a different CV stack.
    """

    view: ViewKind
    head_visible: bool
    feet_visible: bool
    subject_frame_fraction: float
    sharpness: float
    exposure: float
    background_separability: float
    arm_separation: float
    torso_verticality: float
    device_pitch_deg: float
    clothing_tightness: float

    def __post_init__(self) -> None:
        for name in (
            "subject_frame_fraction", "sharpness", "exposure", "background_separability",
            "arm_separation", "torso_verticality", "clothing_tightness",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a normalised score in [0, 1], got {value!r}")
        if not math.isfinite(self.device_pitch_deg):
            raise ValueError("device_pitch_deg must be finite")


@dataclass(frozen=True, slots=True)
class GateVerdict:
    gate_id: str
    passed: bool
    score: float
    remediation: RemediationCode | None

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score!r}")
        if not self.passed and self.remediation is None:
            raise ValueError(
                f"gate {self.gate_id!r} failed without a remediation; a rejection the user "
                "cannot act on is a dead end"
            )
        if self.passed and self.remediation is not None:
            raise ValueError(f"gate {self.gate_id!r} passed but carries a remediation")


@dataclass(frozen=True, slots=True)
class CaptureBundle:
    capture_id: str
    frontal: PhotoRef
    lateral: PhotoRef
    declared_height: Measure
    declared_weight: Mass | None
    device: DeviceMetadata
    gate_report: tuple[GateVerdict, ...]

    def __post_init__(self) -> None:
        if not self.gate_report:
            raise ValueError("gate_report must not be empty; an ungated capture is not validated")
        failed = [v.gate_id for v in self.gate_report if not v.passed]
        if failed:
            raise ValueError(
                f"cannot build a CaptureBundle from failed gates: {', '.join(failed)}"
            )
        if self.frontal == self.lateral:
            raise ValueError("frontal and lateral must be distinct photographs")
        if not isinstance(self.declared_height, Measure):
            raise TypeError(
                "declared_height must be a Measure (centimetres), got "
                f"{type(self.declared_height).__name__}"
            )
        if self.declared_weight is not None and not isinstance(self.declared_weight, Mass):
            raise TypeError(
                "declared_weight must be a Mass (kilograms), not a "
                f"{type(self.declared_weight).__name__}. Mass and Measure have the same shape, "
                "so this is the one place the confusion is both plausible and silent."
            )
        if self.declared_height.source is not MeasureSource.USER_DECLARED:
            raise ValueError(
                f"declared_height must have source USER_DECLARED, got {self.declared_height.source}"
            )
        if not _MIN_PLAUSIBLE_HEIGHT_CM <= self.declared_height.value_cm <= _MAX_PLAUSIBLE_HEIGHT_CM:
            raise ValueError(
                f"declared_height of {self.declared_height.value_cm} cm is outside the plausible "
                f"range [{_MIN_PLAUSIBLE_HEIGHT_CM}, {_MAX_PLAUSIBLE_HEIGHT_CM}]"
            )
        if (
            self.declared_weight is not None
            and self.declared_weight.source is not MeasureSource.USER_DECLARED
        ):
            raise ValueError(
                f"declared_weight must have source USER_DECLARED, got {self.declared_weight.source}"
            )
        object.__setattr__(self, "gate_report", tuple(self.gate_report))
