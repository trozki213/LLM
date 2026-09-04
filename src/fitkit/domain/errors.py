"""The error taxonomy. Callers branch on these categories, so they are a contract.

Two rules from the design govern this module:

* No foreign exception crosses a boundary -- adapters translate into these types.
* Fail closed on numbers, fail open on prose. Degradation is therefore a *flag*
  (`DegradationCode`), never an exception: a missing explanation must not fail a request.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, types only
    from fitkit.domain.regions import BodyRegion


class FitKitError(Exception):
    """Root of every error this system raises deliberately."""


class InputError(FitKitError):
    """The caller or the user can fix this. Maps to 4xx."""


class InfrastructureError(FitKitError):
    """We cannot fix this in the request. Maps to 5xx.

    Named `InfrastructureError` rather than `SystemError` so it does not shadow the
    builtin of that name in any module that imports it.
    """


class CaptureRejected(InputError):
    def __init__(self, gate_ids: tuple[str, ...]) -> None:
        self.gate_ids = tuple(gate_ids)
        super().__init__(f"capture rejected by gates: {', '.join(self.gate_ids)}")


class InvalidDeclaredHeight(InputError):
    pass


class GarmentNotFound(InputError):
    pass


class SizeNotFound(InputError):
    pass


class SizeSpecIncomplete(InputError):
    pass


class MissingRegion(InputError):
    def __init__(self, region: "BodyRegion") -> None:
        self.region = region
        super().__init__(f"no measurement for body region {region.name}")


class UnsupportedSchemaVersion(InputError):
    def __init__(self, found: str | None) -> None:
        self.found = found
        super().__init__(f"unsupported contract schema_version: {found!r}")


class BackendUnavailable(InfrastructureError):
    pass


class BackendTimeout(InfrastructureError):
    pass


class StorageError(InfrastructureError):
    pass


class UncalibratedBackend(InfrastructureError):
    """A backend has no measured residuals, so we cannot state its uncertainty honestly.

    Failing here is the point: ADR-010 says uncertainty is measured, not asserted, and a
    backend that has not been characterised on the validation panel must not ship.
    """


class ContractViolation(InfrastructureError):
    """A document failed its own contract. Always fail closed on this."""

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        super().__init__(f"contract violation at {field!r}: {reason}")


class DegradationCode(StrEnum):
    """Successful results that are worth less than a full one. Not exceptions."""

    MEASUREMENT_UNCERTAIN = "measurement_uncertain"
    COVERAGE_PARTIAL = "coverage_partial"
    EXPLANATION_TEMPLATED = "explanation_templated"
