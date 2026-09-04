"""Acquisition: gates are pure functions over signals, and rejections are actionable."""
import pytest

from fitkit.acquisition import (
    CaptureAssembler,
    CompositeGate,
    DeviceTiltGate,
    DistanceGate,
    FramingGate,
    RawCapture,
    ThresholdGate,
    standard_gates,
)
from fitkit.domain.capture import DeviceMetadata, FrameSignals, GateVerdict, RemediationCode, ViewKind
from fitkit.domain.errors import CaptureRejected
from fitkit.domain.units import MeasureSource

from tests.fakes import InMemoryPhotoStore

DEVICE = DeviceMetadata(platform="ios", model="iPhone15,2", app_version="1.0.0")


def signals(view=ViewKind.FRONTAL, **overrides) -> FrameSignals:
    fields = dict(
        view=view, head_visible=True, feet_visible=True, subject_frame_fraction=0.78,
        sharpness=0.9, exposure=0.7, background_separability=0.8, arm_separation=0.8,
        torso_verticality=0.95, device_pitch_deg=2.0, clothing_tightness=0.8,
    )
    fields.update(overrides)
    return FrameSignals(**fields)


class StubAnalyzer:
    analyzer_id = "stub/1"

    def __init__(self, **overrides) -> None:
        self._overrides = overrides

    def analyze(self, image: bytes, view: ViewKind) -> FrameSignals:
        return signals(view=view, **self._overrides)


class TestIndividualGates:
    def test_framing_passes_a_full_body(self):
        assert FramingGate().evaluate(signals()).passed

    def test_framing_rejects_cropped_feet_with_a_remediation(self):
        v = FramingGate().evaluate(signals(feet_visible=False))
        assert not v.passed
        assert v.remediation is RemediationCode.FULL_BODY_IN_FRAME

    def test_distance_tells_you_which_way_to_move(self):
        assert DistanceGate().evaluate(signals(subject_frame_fraction=0.3)).remediation is (
            RemediationCode.STEP_CLOSER
        )
        assert DistanceGate().evaluate(signals(subject_frame_fraction=0.99)).remediation is (
            RemediationCode.STEP_BACK
        )

    def test_a_threshold_gate_reports_the_signal_as_its_score(self):
        gate = ThresholdGate("sharpness", "sharpness", 0.6, RemediationCode.HOLD_STILL)
        assert gate.evaluate(signals(sharpness=0.42)).score == pytest.approx(0.42)

    def test_device_tilt_reports_a_normalised_score(self):
        v = DeviceTiltGate().evaluate(signals(device_pitch_deg=30.0))
        assert not v.passed
        assert 0.0 <= v.score <= 1.0

    def test_device_tilt_is_symmetric(self):
        assert (
            DeviceTiltGate().evaluate(signals(device_pitch_deg=-30.0)).passed
            is DeviceTiltGate().evaluate(signals(device_pitch_deg=30.0)).passed
        )


class TestCompositeGate:
    def test_reports_every_child_verdict(self):
        composite = standard_gates(ViewKind.FRONTAL)
        assert len(composite.evaluate_all(signals())) == 9

    def test_the_lateral_protocol_omits_the_arm_gate(self):
        ids = {v.gate_id for v in standard_gates(ViewKind.LATERAL).evaluate_all(signals())}
        assert "arm_separation" not in ids

    def test_passes_only_when_every_child_passes(self):
        assert standard_gates(ViewKind.FRONTAL).evaluate(signals()).passed

    def test_surfaces_all_problems_at_once_not_just_the_first(self):
        bad = signals(sharpness=0.1, exposure=0.1, feet_visible=False)
        failures = [v for v in standard_gates(ViewKind.FRONTAL).evaluate_all(bad) if not v.passed]
        assert len(failures) == 3

    def test_reports_the_most_actionable_remediation_first(self):
        bad = signals(sharpness=0.1, feet_visible=False)
        assert standard_gates(ViewKind.FRONTAL).evaluate(bad).remediation is (
            RemediationCode.FULL_BODY_IN_FRAME
        )

    def test_an_empty_composite_is_refused(self):
        with pytest.raises(ValueError, match="at least one child"):
            CompositeGate("empty", ())

    def test_composition_is_testable_with_stub_children(self):
        class AlwaysFails:
            gate_id = "stub"

            def evaluate(self, signals):
                return GateVerdict("stub", False, 0.0, RemediationCode.MORE_LIGHT)

        composite = CompositeGate("c", (AlwaysFails(), AlwaysFails()))
        assert not composite.evaluate(signals()).passed


class TestEveryRejectionIsActionable:
    def test_no_gate_can_fail_without_saying_how_to_fix_it(self):
        """A rejection with no remediation is a dead end, so it cannot be represented."""
        broken = signals(
            head_visible=False, feet_visible=False, subject_frame_fraction=0.2,
            sharpness=0.0, exposure=0.0, background_separability=0.0,
            arm_separation=0.0, torso_verticality=0.0, device_pitch_deg=45.0,
            clothing_tightness=0.0,
        )
        for view in ViewKind:
            for verdict in standard_gates(view).evaluate_all(broken):
                assert verdict.passed or verdict.remediation is not None


class TestCaptureAssembler:
    def _assembler(self, **overrides):
        return CaptureAssembler(StubAnalyzer(**overrides), InMemoryPhotoStore())

    def _raw(self, **overrides):
        fields = dict(
            frontal=b"frontal-bytes", lateral=b"lateral-bytes",
            declared_height_cm=175.0, declared_weight_kg=None, device=DEVICE,
        )
        fields.update(overrides)
        return RawCapture(**fields)

    def test_builds_a_bundle_from_a_clean_capture(self):
        bundle = self._assembler().assemble("cap_1", self._raw())
        assert bundle.capture_id == "cap_1"
        assert bundle.declared_height.source is MeasureSource.USER_DECLARED
        assert len(bundle.gate_report) == 17  # 9 frontal + 8 lateral

    def test_declared_height_is_never_certain(self):
        bundle = self._assembler().assemble("cap_1", self._raw())
        assert bundle.declared_height.sigma_cm > 0

    def test_a_failed_gate_rejects_the_capture(self):
        with pytest.raises(CaptureRejected) as exc:
            self._assembler(sharpness=0.1).assemble("cap_1", self._raw())
        assert any("sharpness" in gate for gate in exc.value.gate_ids)

    def test_the_rejection_names_the_view_that_failed(self):
        with pytest.raises(CaptureRejected) as exc:
            self._assembler(sharpness=0.1).assemble("cap_1", self._raw())
        assert any(gate.startswith("frontal.") for gate in exc.value.gate_ids)

    def test_a_rejected_capture_stores_no_photographs(self):
        photos = InMemoryPhotoStore()
        assembler = CaptureAssembler(StubAnalyzer(sharpness=0.1), photos)
        with pytest.raises(CaptureRejected):
            assembler.assemble("cap_1", self._raw())
        assert photos._blobs == {}

    def test_weight_is_carried_as_a_mass_when_supplied(self):
        bundle = self._assembler().assemble("cap_1", self._raw(declared_weight_kg=72.0))
        assert bundle.declared_weight.value_kg == 72.0
        assert bundle.declared_weight.sigma_kg > 0

    def test_the_two_views_are_stored_separately(self):
        bundle = self._assembler().assemble("cap_1", self._raw())
        assert bundle.frontal != bundle.lateral

    def test_an_implausible_height_is_refused_by_the_domain(self):
        with pytest.raises(ValueError, match="declared_height"):
            self._assembler().assemble("cap_1", self._raw(declared_height_cm=15.0))
