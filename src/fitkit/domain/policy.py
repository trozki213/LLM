"""Merchant-tunable policy. Thresholds are inputs, not constants.

Abstention has a business cost, and the right operating point differs between a merchant
with free returns and one making to order (design 7.2). So the thresholds that decide
SINGLE / TWO_SIZES / ABSTAIN live here, versioned, and are recorded in every assessment.

bare-cm-exempt: ease bounds and the sigma ceiling are policy constants -- exact by
definition, because they are the rule being applied rather than a measurement of a body.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from fitkit.domain.regions import BodyRegion


class FitPreference(StrEnum):
    """What the shopper asked for, relative to the designer's intent."""

    TIGHTER = "tighter"
    AS_DESIGNED = "as_designed"
    LOOSER = "looser"


class Tone(StrEnum):
    """A render hint. It never changes a number, only how one is said."""

    NEUTRAL = "neutral"
    WARM = "warm"


@dataclass(frozen=True, slots=True)
class EaseWindow:
    """Acceptable ease for one region: the least, the ideal, and the most."""

    min_cm: float
    preferred_cm: float
    max_cm: float

    def __post_init__(self) -> None:
        values = (self.min_cm, self.preferred_cm, self.max_cm)
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"ease bounds must be finite, got {values!r}")
        if not self.min_cm <= self.preferred_cm <= self.max_cm:
            raise ValueError(f"ease bounds must be ordered min <= preferred <= max, got {values!r}")

    def contains(self, delta_cm: float) -> bool:
        return self.min_cm <= delta_cm <= self.max_cm


@dataclass(frozen=True)
class FitPolicy:
    """One merchant's risk appetite and region priorities, at one immutable version."""

    policy_id: str
    version: int
    tau_single: float
    tau_pair: float
    max_critical_sigma_cm: float
    region_weights: Mapping[BodyRegion, float]
    critical_regions: frozenset[BodyRegion]
    tightness_penalty_ratio: float

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError(f"version must be >= 1, got {self.version!r}")
        if not 0.0 < self.tau_single <= 1.0:
            raise ValueError(f"tau_single must be in (0, 1], got {self.tau_single!r}")
        if not self.tau_single <= self.tau_pair <= 1.0:
            raise ValueError(
                f"tau_pair must be in [tau_single, 1], got {self.tau_pair!r}; offering two "
                "sizes must not be easier than committing to one"
            )
        if not math.isfinite(self.max_critical_sigma_cm) or self.max_critical_sigma_cm <= 0:
            raise ValueError(
                f"max_critical_sigma_cm must be finite and > 0, got {self.max_critical_sigma_cm!r}"
            )
        if not self.region_weights:
            raise ValueError("region_weights must name at least one region")
        for region, weight in self.region_weights.items():
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"region_weights[{region.name}] must be > 0, got {weight!r}")
        unweighted = self.critical_regions - set(self.region_weights)
        if unweighted:
            raise ValueError(
                "critical regions must be weighted: "
                f"{', '.join(sorted(r.name for r in unweighted))}"
            )
        if self.tightness_penalty_ratio < 1.0:
            raise ValueError(
                f"tightness_penalty_ratio must be >= 1, got {self.tightness_penalty_ratio!r}; "
                "a value below 1 would make a too-tight garment score better than a too-loose one"
            )
        object.__setattr__(self, "region_weights", MappingProxyType(dict(self.region_weights)))
        object.__setattr__(self, "critical_regions", frozenset(self.critical_regions))

    @property
    def version_key(self) -> str:
        return f"{self.policy_id}/{self.version}"
