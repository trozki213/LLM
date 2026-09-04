"""Ports are the seams. These tests fail when a port's surface drifts silently."""
import inspect
import typing

import pytest

from fitkit.domain import ports


EXPECTED = {
    "FrameAnalyzer": {"analyze"},
    "HttpTransport": {"post_json"},
    "MeasurementBackend": {"estimate"},
    "ScaleCalibrationSource": {"calibrate"},
    "GarmentRepository": {"get", "latest_version"},
    "ExplanationRenderer": {"render"},
    "LlmClient": {"complete"},
    "PhotoStore": {"put", "get", "delete"},
    "AssessmentStore": {"save", "load"},
    "Clock": {"now"},
    "MetricsPort": {"increment", "observe"},
}


def _protocols():
    return {
        name: obj
        for name, obj in vars(ports).items()
        if inspect.isclass(obj) and getattr(obj, "_is_protocol", False)
    }


class TestPortSurface:
    def test_every_expected_port_exists(self):
        assert set(EXPECTED) <= set(_protocols())

    def test_no_undeclared_port_has_appeared(self):
        """A new seam is an architectural decision; it should not arrive unnoticed."""
        assert set(_protocols()) == set(EXPECTED)

    @pytest.mark.parametrize("name,methods", sorted((k, tuple(sorted(v))) for k, v in EXPECTED.items()))
    def test_port_declares_exactly_its_methods(self, name, methods):
        proto = _protocols()[name]
        declared = {
            m for m, v in vars(proto).items()
            if callable(v) and not m.startswith("_")
        }
        assert declared == set(methods)

    def test_every_port_method_is_fully_annotated(self):
        for name, proto in _protocols().items():
            for method_name, method in vars(proto).items():
                if not callable(method) or method_name.startswith("_"):
                    continue
                hints = typing.get_type_hints(method)
                params = [
                    p for p in inspect.signature(method).parameters if p not in ("self", "cls")
                ]
                assert "return" in hints, f"{name}.{method_name} has no return annotation"
                for p in params:
                    assert p in hints, f"{name}.{method_name}({p}) is unannotated"


class TestStructuralSubstitutability:
    def test_a_plain_object_satisfies_a_port_without_inheriting_from_it(self):
        """Protocols, not base classes: an adapter never imports a framework to comply."""
        import datetime as dt

        class FakeClock:
            def now(self) -> dt.datetime:
                return dt.datetime(2026, 9, 3, tzinfo=dt.UTC)

        clock: ports.Clock = FakeClock()
        assert clock.now().year == 2026

    def test_ports_are_not_runtime_checkable_by_accident(self):
        """isinstance against a Protocol checks names only; it is a false comfort."""
        for name, proto in _protocols().items():
            assert not getattr(proto, "_is_runtime_protocol", False), name


def assert_conforms(candidate: object, protocol: type) -> None:
    """Structural conformance, checked at runtime.

    Protocols are not runtime_checkable here on purpose -- isinstance against one checks
    names only and is a false comfort. This checks names *and* parameter lists, which is
    what actually breaks when a port drifts.
    """
    for name, member in vars(protocol).items():
        if not callable(member) or name.startswith("_"):
            continue
        impl = getattr(candidate, name, None)
        assert callable(impl), f"{type(candidate).__name__} is missing {protocol.__name__}.{name}"
        expected = [p for p in inspect.signature(member).parameters if p not in ("self", "cls")]
        actual = [p for p in inspect.signature(impl).parameters if p not in ("self", "cls")]
        assert actual == expected, (
            f"{type(candidate).__name__}.{name}{tuple(actual)} does not match "
            f"{protocol.__name__}.{name}{tuple(expected)}"
        )
    for attr, hint in getattr(protocol, "__annotations__", {}).items():
        assert hasattr(candidate, attr), (
            f"{type(candidate).__name__} is missing the {protocol.__name__}.{attr} attribute"
        )


class TestFakesConformToTheirPorts:
    """The doubles Phases 2 and 5 will build on, checked against the seams they stand in for."""

    def test_every_fake_matches_its_port(self):
        from tests import fakes

        pairs = [
            (fakes.FixedMeasurementBackend(), ports.MeasurementBackend),
            (fakes.PerturbingMeasurementBackend(residual_cm=3.0), ports.MeasurementBackend),
            (fakes.FailingMeasurementBackend(), ports.MeasurementBackend),
            (fakes.DeclaredHeightCalibrationStub(), ports.ScaleCalibrationSource),
            (fakes.InMemoryGarmentRepository(), ports.GarmentRepository),
            (fakes.StubRenderer(), ports.ExplanationRenderer),
            (fakes.ScriptedLlmClient(), ports.LlmClient),
            (fakes.InMemoryPhotoStore(), ports.PhotoStore),
            (fakes.InMemoryAssessmentStore(), ports.AssessmentStore),
            (fakes.RecordingMetrics(), ports.MetricsPort),
        ]
        for fake, protocol in pairs:
            assert_conforms(fake, protocol)

    def test_a_drifted_implementation_is_caught(self):
        """Guards the guard: a wrong signature must actually fail."""

        class WrongClock:
            def now(self, tz: str) -> None: ...

        with pytest.raises(AssertionError, match="does not match"):
            assert_conforms(WrongClock(), ports.Clock)

    def test_a_missing_attribute_is_caught(self):
        class BackendWithoutAnId:
            supported_regions = frozenset()

            def estimate(self, bundle, calibration): ...

        with pytest.raises(AssertionError, match="backend_id"):
            assert_conforms(BackendWithoutAnId(), ports.MeasurementBackend)
