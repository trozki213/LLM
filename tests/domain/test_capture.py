"""A CaptureBundle is the *validated* output of acquisition. Invalid ones cannot exist."""
import pytest

from fitkit.domain.capture import (
    CaptureBundle,
    DeviceMetadata,
    GateVerdict,
    PhotoRef,
    RemediationCode,
    ViewKind,
)
from fitkit.domain.units import Mass, Measure, MeasureSource

PHOTO = PhotoRef(uri="s3://captures/cap_01J/frontal.jpg", sha256="a" * 64)
LATERAL = PhotoRef(uri="s3://captures/cap_01J/lateral.jpg", sha256="b" * 64)
DEVICE = DeviceMetadata(platform="ios", model="iPhone15,2", app_version="1.0.0")
PASSED = (GateVerdict(gate_id="framing", passed=True, score=0.94, remediation=None),)


def bundle(**overrides) -> CaptureBundle:
    fields = dict(
        capture_id="cap_01J",
        frontal=PHOTO,
        lateral=LATERAL,
        declared_height=Measure(175.0, 1.5, MeasureSource.USER_DECLARED),
        declared_weight=None,
        device=DEVICE,
        gate_report=PASSED,
    )
    fields.update(overrides)
    return CaptureBundle(**fields)


class TestValidity:
    def test_a_failed_gate_makes_the_bundle_unconstructible(self):
        failed = GateVerdict(
            gate_id="framing", passed=False, score=0.2, remediation=RemediationCode.STEP_BACK
        )
        with pytest.raises(ValueError, match="framing"):
            bundle(gate_report=PASSED + (failed,))

    def test_requires_a_gate_report(self):
        with pytest.raises(ValueError, match="gate_report"):
            bundle(gate_report=())

    def test_declared_height_must_be_declared_by_the_user(self):
        with pytest.raises(ValueError, match="declared_height"):
            bundle(declared_height=Measure(175.0, 1.5, MeasureSource.ESTIMATED))

    def test_declared_height_cannot_be_certain(self):
        """Design 7.4: self-reported height is a first-class *uncertain* input."""
        with pytest.raises(ValueError, match="sigma_cm"):
            bundle(declared_height=Measure(175.0, 0.0, MeasureSource.USER_DECLARED))

    def test_implausible_height_is_rejected(self):
        with pytest.raises(ValueError, match="declared_height"):
            bundle(declared_height=Measure(15.0, 1.5, MeasureSource.USER_DECLARED))

    def test_weight_is_optional(self):
        assert bundle().declared_weight is None

    def test_weight_is_a_mass_not_a_length(self):
        """Kilograms must not travel in a type whose field is called value_cm."""
        with pytest.raises((ValueError, TypeError)):
            bundle(declared_weight=Measure(72.0, 2.0, MeasureSource.USER_DECLARED))
        assert bundle(
            declared_weight=Mass(72.0, 2.0, MeasureSource.USER_DECLARED)
        ).declared_weight.value_kg == 72.0

    def test_weight_when_present_must_be_declared_by_the_user(self):
        with pytest.raises(ValueError, match="declared_weight"):
            bundle(declared_weight=Mass(72.0, 2.0, MeasureSource.ESTIMATED))

    def test_the_two_views_must_be_different_photographs(self):
        with pytest.raises(ValueError, match="distinct"):
            bundle(lateral=PHOTO)


class TestGateVerdict:
    def test_a_failed_gate_must_say_how_to_fix_it(self):
        """A rejection with no remediation is a dead end for the user."""
        with pytest.raises(ValueError, match="remediation"):
            GateVerdict(gate_id="blur", passed=False, score=0.1, remediation=None)

    def test_a_passed_gate_carries_no_remediation(self):
        with pytest.raises(ValueError, match="remediation"):
            GateVerdict(gate_id="blur", passed=True, score=0.9, remediation=RemediationCode.HOLD_STILL)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_score_is_normalised(self, bad):
        with pytest.raises(ValueError, match="score"):
            GateVerdict(gate_id="blur", passed=True, score=bad, remediation=None)

    def test_every_remediation_code_is_actionable_text_downstream(self):
        assert len(RemediationCode) >= 5
        assert all(code.value.islower() for code in RemediationCode)


class TestViewKind:
    def test_the_protocol_is_two_views(self):
        assert set(ViewKind) == {ViewKind.FRONTAL, ViewKind.LATERAL}
