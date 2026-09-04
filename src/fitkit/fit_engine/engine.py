"""The deterministic fit engine.

C1: every comparison, threshold, ranking and size choice happens here, in code that has
no clock, no randomness, no network and no natural language. The only thing that leaves
is a `FitAssessment` document.

Purity note: `assessment_id` and `computed_at` are parameters rather than something the
engine generates. The contract requires both, and a pure function cannot invent either,
so the impurity is pushed onto the caller where it is visible.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import datetime as dt

from fitkit.domain.body import BodyMeasurements
from fitkit.domain.contracts.fit_assessment import (
    AbstainReason,
    Coverage,
    FabricSummary,
    FitAssessment,
    GarmentRef,
    InputsDigest,
    Recommendation,
    RegionDelta,
    RenderHints,
    SizeAssessment,
    SizeChoice,
    Verdict,
)
from fitkit.domain.garment import GarmentSizeSpec, GarmentSpec
from fitkit.domain.policy import EaseWindow, FitPolicy, FitPreference, Tone
from fitkit.domain.regions import BodyRegion, GarmentRegion
from fitkit.domain.units import MIN_SIGMA_CM, Measure
from fitkit.fit_engine.abstain import AbstainPolicy, Decision, RankedSize, ThresholdAbstainPolicy
from fitkit.fit_engine.ease import ConventionalEaseRules, EaseRulePolicy
from fitkit.fit_engine.quadrature import RESIDUAL_NODES, SCALE_NODES
from fitkit.fit_engine.scoring import classify, is_uncertain, penalty
from fitkit.fit_engine.stretch import ClassBasedStretchModel, StretchModel

ENGINE_VERSION: Final[str] = "fit-engine/1.0.0"

#: Emitted centimetres are rounded here so the serialised document is stable.
_CM_DP: Final[int] = 2

#: Regions given their own residual grid. Beyond this the product grid grows faster than
#: it buys accuracy, so lower-weighted regions are held at their mean.
_MAX_PRODUCT_REGIONS: Final[int] = 5

#: Score gap below which two sizes are treated as tied at a node, and split the weight.
_TIE_EPSILON: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class _RegionComparison:
    region: BodyRegion
    raw_delta_cm: float
    delta_sigma_cm: float
    residual_sigma_cm: float
    body_value_cm: float
    stretch_capacity_cm: float
    window: EaseWindow
    critical: bool
    weight: float

    @property
    def delta_cm(self) -> float:
        return _effective_delta(self.raw_delta_cm, self.stretch_capacity_cm)

    @property
    def stretch_absorbed_cm(self) -> float:
        return self.delta_cm - self.raw_delta_cm


def _effective_delta(raw_delta_cm: float, capacity_cm: float) -> float:
    """Stretch absorbs a shortfall; it does not add room to a garment that already fits.

    Applying the full extension unconditionally would score an elastane garment as
    *looser* than the same measurements in denim, which is backwards: relaxed, they are
    the same size. The stretch only buys you something when you need it.
    """
    if raw_delta_cm >= 0.0:
        return raw_delta_cm
    return raw_delta_cm + min(-raw_delta_cm, capacity_cm)


class DeterministicFitEngine:
    """Composition of three strategies plus fixed-node quadrature. No inheritance."""

    def __init__(
        self,
        *,
        ease_rules: EaseRulePolicy | None = None,
        stretch_model: StretchModel | None = None,
        abstain_policy: AbstainPolicy | None = None,
    ) -> None:
        self._ease = ease_rules or ConventionalEaseRules()
        self._stretch = stretch_model or ClassBasedStretchModel()
        self._abstain = abstain_policy or ThresholdAbstainPolicy()

    @property
    def engine_version(self) -> str:
        return f"{ENGINE_VERSION}+{self._ease.rules_id}+{self._stretch.model_id}"

    def assess(
        self,
        *,
        body: BodyMeasurements,
        garment: GarmentSpec,
        preference: FitPreference,
        policy: FitPolicy,
        assessment_id: str,
        computed_at: dt.datetime,
        locale: str = "en",
        tone: Tone = Tone.NEUTRAL,
    ) -> FitAssessment:
        comparable, missing = self._coverage(body, garment, policy)
        coverage = Coverage.COMPLETE if not missing else Coverage.PARTIAL

        per_size = {
            size.size_label: self._compare_size(body, garment, size, preference, policy, comparable)
            for size in garment.sizes
        }
        probabilities = self._probabilities(per_size, body, policy)

        sizes = tuple(
            SizeAssessment(
                size_label=label,
                confidence=probabilities[label],
                regions=tuple(_to_delta(c) for c in comparisons),
                coverage=coverage,
                missing_regions=missing,
            )
            for label, comparisons in per_size.items()
        )
        ranked = tuple(
            sorted(
                (RankedSize(s, s.confidence) for s in sizes),
                key=lambda r: (-r.probability, r.assessment.size_label),
            )
        )
        decision = self._abstain.decide(ranked, body, policy)

        return FitAssessment(
            assessment_id=assessment_id,
            garment=GarmentRef(
                garment_id=garment.garment_id,
                category=garment.category,
                size_system=garment.size_system,
                fit_intent=garment.fit_intent,
            ),
            fabric=FabricSummary(
                stretch_class=garment.fabric.stretch_class,
                recovery=garment.fabric.recovery,
                usable_extension_pct=round(
                    100.0
                    * self._stretch.usable_extension(
                        garment.fabric, BodyRegion.WAIST, garment.category
                    ),
                    _CM_DP,
                ),
            ),
            recommendation=_recommend(decision, ranked),
            sizes=sizes,
            inputs_digest=InputsDigest(
                measurement_backend=(
                    f"{body.provenance.backend_id}@{body.provenance.backend_version}"
                ),
                measurement_provenance_id=body.provenance.capture_id,
                garment_spec_version=garment.version_key,
                engine_version=self.engine_version,
                policy_version=policy.version_key,
                residual_table_version=body.provenance.residual_table_version,
                computed_at=computed_at,
            ),
            render_hints=RenderHints(locale=locale, tone=tone),
        )

    # -- internals ---------------------------------------------------------------

    def _coverage(
        self, body: BodyMeasurements, garment: GarmentSpec, policy: FitPolicy
    ) -> tuple[tuple[GarmentRegion, ...], tuple[BodyRegion, ...]]:
        spec_regions = garment.sizes[0].regions
        comparable = tuple(
            sorted(
                (
                    gr
                    for gr in spec_regions
                    if gr.body_region in body.regions and gr.body_region in policy.region_weights
                ),
                key=lambda gr: gr.name,
            )
        )
        covered = {gr.body_region for gr in comparable}
        missing = tuple(sorted(set(policy.region_weights) - covered, key=lambda r: r.name))
        return comparable, missing

    def _compare_size(
        self,
        body: BodyMeasurements,
        garment: GarmentSpec,
        size: GarmentSizeSpec,
        preference: FitPreference,
        policy: FitPolicy,
        comparable: Sequence[GarmentRegion],
    ) -> tuple[_RegionComparison, ...]:
        out = []
        for garment_region in comparable:
            region = garment_region.body_region
            spec = size.measurements[garment_region]
            girth = spec.scaled(2.0) if garment_region.is_flat else spec
            extension = self._stretch.usable_extension(garment.fabric, region, garment.category)
            measured = body[region]
            delta = girth - measured
            out.append(
                _RegionComparison(
                    region=region,
                    raw_delta_cm=delta.value_cm,
                    delta_sigma_cm=max(delta.sigma_cm, MIN_SIGMA_CM),
                    residual_sigma_cm=math.hypot(body.residual_sigma_cm(region), girth.sigma_cm),
                    body_value_cm=measured.value_cm,
                    stretch_capacity_cm=extension * girth.value_cm,
                    window=self._ease.required_ease(
                        region, garment.category, garment.fit_intent, preference
                    ),
                    critical=region in policy.critical_regions,
                    weight=policy.region_weights[region],
                )
            )
        return tuple(out)

    def _probabilities(
        self,
        per_size: Mapping[str, tuple[_RegionComparison, ...]],
        body: BodyMeasurements,
        policy: FitPolicy,
    ) -> dict[str, float]:
        """P(size s scores best), integrated over every source of uncertainty.

        The grid is the product of the shared scale nodes with a per-region grid over
        the independent residuals, and at each node the best size simply wins its
        weight. There is no softmin and no temperature: an exact argmin over a fixed
        grid needs no free parameter, and a free parameter here would be a dial nobody
        could justify to a merchant.
        """
        labels = list(per_size)
        if not labels or not any(per_size.values()):
            return {label: 0.0 for label in labels}

        totals = {label: 0.0 for label in labels}
        grids = {label: _residual_grid(per_size[label]) for label in labels}
        for z, scale_weight in SCALE_NODES:
            for offsets, inner_weight in grids[labels[0]]:
                scores = {
                    label: _score_at_node(
                        per_size[label], z, body.scale_sigma_rel, policy, offsets
                    )
                    for label in labels
                }
                best = min(scores.values())
                winners = [label for label in labels if scores[label] <= best + _TIE_EPSILON]
                share = scale_weight * inner_weight / len(winners)
                for label in winners:
                    totals[label] += share
        total = sum(totals.values())
        return {label: round(totals[label] / total, 4) for label in labels}


def _residual_grid(
    comparisons: Sequence[_RegionComparison],
) -> tuple[tuple[tuple[float, ...], float], ...]:
    """The product grid over per-region independent residuals, with a size cap.

    Beyond `_MAX_PRODUCT_REGIONS` the product would grow faster than it buys accuracy,
    so the lowest-weighted regions are held at their mean. The cap is applied by weight
    and then by region name, so the grid is identical for identical inputs.
    """
    ranked = sorted(
        range(len(comparisons)),
        key=lambda i: (-comparisons[i].weight, comparisons[i].region.name),
    )
    gridded = set(ranked[:_MAX_PRODUCT_REGIONS])
    per_region = [
        RESIDUAL_NODES if i in gridded else ((0.0, 1.0),) for i in range(len(comparisons))
    ]
    grid = []
    for combo in itertools.product(*per_region):
        offsets = tuple(u for u, _ in combo)
        weight = 1.0
        for _, w in combo:
            weight *= w
        grid.append((offsets, weight))
    return tuple(grid)


def _score_at_node(
    comparisons: Sequence[_RegionComparison],
    z: float,
    scale_sigma_rel: float,
    policy: FitPolicy,
    offsets: Sequence[float],
) -> float:
    """Weighted penalty at one fully specified node of the grid."""
    total = 0.0
    weight_sum = 0.0
    for c, u in zip(comparisons, offsets, strict=True):
        # The body shifts with the shared scale factor, so stretch is re-applied here
        # rather than frozen at the mean.
        shifted_raw = (
            c.raw_delta_cm - z * scale_sigma_rel * c.body_value_cm + u * c.residual_sigma_cm
        )
        total += c.weight * penalty(
            _effective_delta(shifted_raw, c.stretch_capacity_cm),
            c.window,
            policy.tightness_penalty_ratio,
        )
        weight_sum += c.weight
    return total / weight_sum if weight_sum else 0.0


def _to_delta(c: _RegionComparison) -> RegionDelta:
    delta = round(c.delta_cm, _CM_DP)
    sigma = round(c.delta_sigma_cm, _CM_DP)
    return RegionDelta(
        region=c.region,
        critical=c.critical,
        delta_cm=delta,
        delta_sigma_cm=max(sigma, MIN_SIGMA_CM),
        stretch_absorbed_cm=round(max(c.stretch_absorbed_cm, 0.0), _CM_DP),
        required_ease=EaseWindow(
            min_cm=round(c.window.min_cm, _CM_DP),
            preferred_cm=round(c.window.preferred_cm, _CM_DP),
            max_cm=round(c.window.max_cm, _CM_DP),
        ),
        classification=classify(c.delta_cm, c.window),
        uncertain=is_uncertain(c.delta_cm, c.delta_sigma_cm, c.window),
    )


def _recommend(decision: Decision, ranked: Sequence[RankedSize]) -> Recommendation:
    if decision.verdict is Verdict.ABSTAIN:
        return Recommendation(
            verdict=Verdict.ABSTAIN,
            primary=None,
            alternate=None,
            abstain=AbstainReason(decision.abstain_code, decision.detail_codes),
        )
    primary = SizeChoice(ranked[0].assessment.size_label, ranked[0].probability)
    alternate = (
        SizeChoice(ranked[1].assessment.size_label, ranked[1].probability)
        if decision.verdict is Verdict.TWO_SIZES
        else None
    )
    return Recommendation(
        verdict=decision.verdict, primary=primary, alternate=alternate, abstain=None
    )
