"""The error taxonomy is a contract: callers branch on these categories (design 4)."""
import pytest

from fitkit.domain import errors as e
from fitkit.domain.regions import BodyRegion


class TestTaxonomy:
    @pytest.mark.parametrize(
        "cls",
        [e.CaptureRejected, e.InvalidDeclaredHeight, e.GarmentNotFound, e.SizeSpecIncomplete,
         e.MissingRegion, e.UnsupportedSchemaVersion, e.SizeNotFound],
    )
    def test_input_errors_are_input_errors(self, cls):
        assert issubclass(cls, e.InputError)
        assert issubclass(cls, e.FitKitError)

    @pytest.mark.parametrize(
        "cls", [e.BackendUnavailable, e.BackendTimeout, e.StorageError, e.ContractViolation]
    )
    def test_system_errors_are_system_errors(self, cls):
        assert issubclass(cls, e.InfrastructureError)
        assert issubclass(cls, e.FitKitError)

    def test_input_and_system_errors_are_disjoint(self):
        assert not issubclass(e.InputError, e.InfrastructureError)
        assert not issubclass(e.InfrastructureError, e.InputError)

    def test_the_infrastructure_root_does_not_shadow_the_builtin(self):
        assert not hasattr(e, "SystemError") or e.SystemError is SystemError

    def test_every_error_is_catchable_as_one_root(self):
        for name in dir(e):
            obj = getattr(e, name)
            if isinstance(obj, type) and issubclass(obj, Exception) and obj.__module__ == e.__name__:
                assert issubclass(obj, e.FitKitError), name

    def test_the_root_is_not_a_bare_exception_alias(self):
        assert e.FitKitError is not Exception
        assert issubclass(e.FitKitError, Exception)


class TestDegradationIsNotAnError:
    def test_degradation_codes_are_an_enum_not_exceptions(self):
        """Fail closed on numbers, fail open on prose: degradation is a flag, not a raise."""
        assert not isinstance(e.DegradationCode, type) or not issubclass(e.DegradationCode, Exception)
        assert {"MEASUREMENT_UNCERTAIN", "COVERAGE_PARTIAL", "EXPLANATION_TEMPLATED"} <= {
            c.name for c in e.DegradationCode
        }


class TestErrorPayloads:
    def test_capture_rejected_carries_the_failures_that_caused_it(self):
        err = e.CaptureRejected(gate_ids=("framing", "blur"))
        assert err.gate_ids == ("framing", "blur")
        assert "framing" in str(err)

    def test_missing_region_names_the_region(self):
        err = e.MissingRegion(BodyRegion.HIP)
        assert err.region is BodyRegion.HIP
        assert "HIP" in str(err)

    def test_contract_violation_names_the_offending_field(self):
        err = e.ContractViolation("recommendation", "field is required")
        assert err.field == "recommendation"
        assert "recommendation" in str(err)
