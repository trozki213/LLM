"""FitAssessment v1 -- the contract between the fit engine and the explanation layer.

This is the most important artifact in the system. It is what makes C1 and C2 structural
rather than aspirational:

* R1  No free text. Every qualitative statement is a closed enum; the engine never
      emits a sentence.
* R2  Completeness. Every number a renderer may state is present here, so there is
      nothing left to compute.
* R3  Isolation. The explanation layer imports this module and nothing else from the
      project, so it physically cannot reach the arithmetic.
* R4  The numeric allowlist. `numeric_allowlist()` derives, from the document alone, the
      set of magnitudes a renderer is permitted to state. The Phase 5 guard checks LLM
      output against exactly this set and falls back to the template on a violation.

Versioning (design 3.3): minors are additive and optional, so an old reader keeps
working; a major bump is refused outright; an unrecognised enum value fails closed.

bare-cm-exempt is not claimed here. Instead, `TestUncertaintyIsAdjacent` asserts that
every `*_cm` field in this contract has a `*_sigma_cm` sibling or a named, justified
exemption -- a stronger rule than the blanket one applied elsewhere.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Mapping, TypeVar

from fitkit.domain.errors import ContractViolation, UnsupportedSchemaVersion
from fitkit.domain.fabric import RecoveryClass, StretchClass
from fitkit.domain.policy import EaseWindow, Tone
from fitkit.domain.regions import BodyRegion, FitIntent, GarmentCategory

SCHEMA_NAME: Final[str] = "fit-assessment"
SCHEMA_MAJOR: Final[int] = 1
SCHEMA_VERSION: Final[str] = f"{SCHEMA_NAME}/{SCHEMA_MAJOR}.0.0"

#: How close a stated number must be to an allowed one to count as the same number.
ALLOWLIST_TOLERANCE: Final[float] = 0.005

E = TypeVar("E", bound=StrEnum)


class Verdict(StrEnum):
    SINGLE = "SINGLE"
    TWO_SIZES = "TWO_SIZES"
    ABSTAIN = "ABSTAIN"


class FitClassification(StrEnum):
    """The engine's qualitative call. Closed, so the renderer cannot invent a softer one."""

    MUCH_TOO_TIGHT = "MUCH_TOO_TIGHT"
    TIGHT = "TIGHT"
    SNUG = "SNUG"
    AS_INTENDED = "AS_INTENDED"
    RELAXED = "RELAXED"
    LOOSE = "LOOSE"
    MUCH_TOO_LOOSE = "MUCH_TOO_LOOSE"


class AbstainCode(StrEnum):
    UNCERTAINTY_EXCEEDS_SIZE_STEP = "UNCERTAINTY_EXCEEDS_SIZE_STEP"
    NO_SIZE_ACCEPTABLE = "NO_SIZE_ACCEPTABLE"
    INSUFFICIENT_GARMENT_DATA = "INSUFFICIENT_GARMENT_DATA"
    INSUFFICIENT_BODY_DATA = "INSUFFICIENT_BODY_DATA"


class Coverage(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"


@dataclass(frozen=True, slots=True)
class GarmentRef:
    garment_id: str
    category: GarmentCategory
    size_system: str
    fit_intent: FitIntent


@dataclass(frozen=True, slots=True)
class FabricSummary:
    stretch_class: StretchClass
    recovery: RecoveryClass
    usable_extension_pct: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.usable_extension_pct) or self.usable_extension_pct < 0:
            raise ValueError(
                f"usable_extension_pct must be finite and >= 0, got {self.usable_extension_pct!r}"
            )


@dataclass(frozen=True, slots=True)
class RegionDelta:
    """How one part of the garment relates to one part of the body, with its uncertainty."""

    region: BodyRegion
    critical: bool
    delta_cm: float
    delta_sigma_cm: float
    stretch_absorbed_cm: float
    required_ease: EaseWindow
    classification: FitClassification
    uncertain: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.delta_cm):
            raise ValueError(f"delta_cm must be finite, got {self.delta_cm!r}")
        if not math.isfinite(self.delta_sigma_cm) or self.delta_sigma_cm <= 0:
            raise ValueError(
                f"delta_sigma_cm must be finite and > 0, got {self.delta_sigma_cm!r}; a delta "
                "without uncertainty is exactly the false precision C6 forbids"
            )
        if not math.isfinite(self.stretch_absorbed_cm) or self.stretch_absorbed_cm < 0:
            raise ValueError(
                f"stretch_absorbed_cm must be finite and >= 0, got {self.stretch_absorbed_cm!r}"
            )


@dataclass(frozen=True, slots=True)
class SizeChoice:
    size_label: str
    confidence: float

    def __post_init__(self) -> None:
        _check_probability(self.confidence, "confidence")


@dataclass(frozen=True, slots=True)
class SizeAssessment:
    size_label: str
    confidence: float
    regions: tuple[RegionDelta, ...]
    coverage: Coverage
    missing_regions: tuple[BodyRegion, ...]

    def __post_init__(self) -> None:
        _check_probability(self.confidence, "confidence")
        if self.coverage is Coverage.PARTIAL and not self.missing_regions:
            raise ValueError(
                "partial coverage must name its missing_regions; silence about what we did "
                "not check is indistinguishable from having checked it"
            )
        if self.coverage is Coverage.COMPLETE and self.missing_regions:
            raise ValueError("complete coverage must have no missing_regions")
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "missing_regions", tuple(self.missing_regions))


@dataclass(frozen=True, slots=True)
class AbstainReason:
    code: AbstainCode
    detail_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail_codes", tuple(self.detail_codes))


@dataclass(frozen=True, slots=True)
class Recommendation:
    verdict: Verdict
    primary: SizeChoice | None
    alternate: SizeChoice | None
    abstain: AbstainReason | None

    def __post_init__(self) -> None:
        if self.verdict is Verdict.ABSTAIN:
            if self.primary is not None:
                raise ValueError("an ABSTAIN verdict must not carry a primary size")
            if self.abstain is None:
                raise ValueError("an ABSTAIN verdict must carry an abstain reason")
            if self.alternate is not None:
                raise ValueError("an ABSTAIN verdict must not carry an alternate size")
            return
        if self.abstain is not None:
            raise ValueError(f"a {self.verdict} verdict must not carry an abstain reason")
        if self.primary is None:
            raise ValueError(f"a {self.verdict} verdict must carry a primary size")
        if self.verdict is Verdict.SINGLE and self.alternate is not None:
            raise ValueError("a SINGLE verdict must not carry an alternate size")
        if self.verdict is Verdict.TWO_SIZES and self.alternate is None:
            raise ValueError("a TWO_SIZES verdict must carry an alternate size")


@dataclass(frozen=True, slots=True)
class InputsDigest:
    """Exactly what produced this assessment, so Phase 7 can replay it."""

    measurement_backend: str
    measurement_provenance_id: str
    garment_spec_version: str
    engine_version: str
    policy_version: str
    residual_table_version: str
    computed_at: dt.datetime

    def __post_init__(self) -> None:
        if self.computed_at.tzinfo is None or self.computed_at.utcoffset() is None:
            raise ValueError("computed_at must carry a timezone")


@dataclass(frozen=True, slots=True)
class RenderHints:
    locale: str
    tone: Tone


@dataclass(frozen=True, slots=True)
class FitAssessment:
    assessment_id: str
    garment: GarmentRef
    fabric: FabricSummary
    recommendation: Recommendation
    sizes: tuple[SizeAssessment, ...]
    inputs_digest: InputsDigest
    render_hints: RenderHints
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.sizes:
            raise ValueError("an assessment must cover at least one size")
        labels = {s.size_label for s in self.sizes}
        for slot in ("primary", "alternate"):
            choice = getattr(self.recommendation, slot)
            if choice is not None and choice.size_label not in labels:
                raise ValueError(
                    f"recommended {slot} {choice.size_label!r} is not among the assessed sizes "
                    f"{sorted(labels)}"
                )
        object.__setattr__(self, "sizes", tuple(self.sizes))

    # -- serialisation -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "assessment_id": self.assessment_id,
            "garment": {
                "garment_id": self.garment.garment_id,
                "category": self.garment.category.value,
                "size_system": self.garment.size_system,
                "fit_intent": self.garment.fit_intent.value,
            },
            "fabric": {
                "stretch_class": self.fabric.stretch_class.value,
                "recovery": self.fabric.recovery.value,
                "usable_extension_pct": self.fabric.usable_extension_pct,
            },
            "recommendation": {
                "verdict": self.recommendation.verdict.value,
                "primary": _choice_to_dict(self.recommendation.primary),
                "alternate": _choice_to_dict(self.recommendation.alternate),
                "abstain": (
                    None
                    if self.recommendation.abstain is None
                    else {
                        "code": self.recommendation.abstain.code.value,
                        "detail_codes": list(self.recommendation.abstain.detail_codes),
                    }
                ),
            },
            "sizes": [
                {
                    "size_label": s.size_label,
                    "confidence": s.confidence,
                    "coverage": s.coverage.value,
                    "missing_regions": [r.value for r in s.missing_regions],
                    "regions": [
                        {
                            "region": d.region.value,
                            "critical": d.critical,
                            "delta_cm": d.delta_cm,
                            "delta_sigma_cm": d.delta_sigma_cm,
                            "stretch_absorbed_cm": d.stretch_absorbed_cm,
                            "required_ease": {
                                "min_cm": d.required_ease.min_cm,
                                "preferred_cm": d.required_ease.preferred_cm,
                                "max_cm": d.required_ease.max_cm,
                            },
                            "classification": d.classification.value,
                            "uncertain": d.uncertain,
                        }
                        for d in s.regions
                    ],
                }
                for s in self.sizes
            ],
            "inputs_digest": {
                "measurement_backend": self.inputs_digest.measurement_backend,
                "measurement_provenance_id": self.inputs_digest.measurement_provenance_id,
                "garment_spec_version": self.inputs_digest.garment_spec_version,
                "engine_version": self.inputs_digest.engine_version,
                "policy_version": self.inputs_digest.policy_version,
                "residual_table_version": self.inputs_digest.residual_table_version,
                "computed_at": self.inputs_digest.computed_at.isoformat(),
            },
            "render_hints": {
                "locale": self.render_hints.locale,
                "tone": self.render_hints.tone.value,
            },
        }

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> FitAssessment:
        version = doc.get("schema_version")
        _check_version(version)
        garment = _require(doc, "garment")
        fabric = _require(doc, "fabric")
        rec = _require(doc, "recommendation")
        digest = _require(doc, "inputs_digest")
        hints = _require(doc, "render_hints")
        try:
            return cls(
                schema_version=str(version),
                assessment_id=str(_require(doc, "assessment_id")),
                garment=GarmentRef(
                    garment_id=str(_require(garment, "garment_id", "garment.garment_id")),
                    category=_enum(GarmentCategory, garment, "category", "garment.category"),
                    size_system=str(_require(garment, "size_system", "garment.size_system")),
                    fit_intent=_enum(FitIntent, garment, "fit_intent", "garment.fit_intent"),
                ),
                fabric=FabricSummary(
                    stretch_class=_enum(StretchClass, fabric, "stretch_class", "fabric.stretch_class"),
                    recovery=_enum(RecoveryClass, fabric, "recovery", "fabric.recovery"),
                    usable_extension_pct=float(
                        _require(fabric, "usable_extension_pct", "fabric.usable_extension_pct")
                    ),
                ),
                recommendation=Recommendation(
                    verdict=_enum(Verdict, rec, "verdict", "recommendation.verdict"),
                    primary=_choice_from_dict(rec.get("primary"), "recommendation.primary"),
                    alternate=_choice_from_dict(rec.get("alternate"), "recommendation.alternate"),
                    abstain=_abstain_from_dict(rec.get("abstain")),
                ),
                sizes=tuple(
                    _size_from_dict(s, i) for i, s in enumerate(_require(doc, "sizes"))
                ),
                inputs_digest=InputsDigest(
                    measurement_backend=str(_require(digest, "measurement_backend", "inputs_digest.measurement_backend")),
                    measurement_provenance_id=str(_require(digest, "measurement_provenance_id", "inputs_digest.measurement_provenance_id")),
                    garment_spec_version=str(_require(digest, "garment_spec_version", "inputs_digest.garment_spec_version")),
                    engine_version=str(_require(digest, "engine_version", "inputs_digest.engine_version")),
                    policy_version=str(_require(digest, "policy_version", "inputs_digest.policy_version")),
                    residual_table_version=str(_require(digest, "residual_table_version", "inputs_digest.residual_table_version")),
                    computed_at=dt.datetime.fromisoformat(
                        str(_require(digest, "computed_at", "inputs_digest.computed_at"))
                    ),
                ),
                render_hints=RenderHints(
                    locale=str(_require(hints, "locale", "render_hints.locale")),
                    tone=_enum(Tone, hints, "tone", "render_hints.tone"),
                ),
            )
        except ValueError as exc:
            if isinstance(exc, (ContractViolation, UnsupportedSchemaVersion)):
                raise
            raise ContractViolation("document", str(exc)) from exc

    # -- the R4 allowlist --------------------------------------------------------

    def numeric_allowlist(self) -> frozenset[float]:
        """Every magnitude a renderer is permitted to state, derived from this document."""
        raw: list[float] = [self.fabric.usable_extension_pct]
        for choice in (self.recommendation.primary, self.recommendation.alternate):
            if choice is not None:
                raw += _from_choice(choice)
        for size in self.sizes:
            raw.append(size.confidence)
            raw.append(size.confidence * 100.0)
            label = _as_number(size.size_label)
            if label is not None:
                raw.append(label)
            for delta in size.regions:
                raw += [
                    delta.delta_cm,
                    delta.delta_sigma_cm,
                    delta.stretch_absorbed_cm,
                    delta.required_ease.min_cm,
                    delta.required_ease.preferred_cm,
                    delta.required_ease.max_cm,
                ]
        allowed: set[float] = set()
        for value in raw:
            for variant in (value, abs(value)):
                allowed.add(float(variant))
                allowed.add(float(round(variant, 1)))
                allowed.add(float(round(variant)))
        return frozenset(allowed)

    def permits(self, value: float, tolerance: float = ALLOWLIST_TOLERANCE) -> bool:
        """True when `value` is a number this document actually states.

        Rebuilds the allowlist on every call. Convenience for one-off checks only: a
        guard scanning many numerals must call `numeric_allowlist()` once and test
        membership against the result itself.
        """
        return any(abs(value - allowed) <= tolerance for allowed in self.numeric_allowlist())


# -- parsing helpers -------------------------------------------------------------


def _check_version(version: Any) -> None:
    if not isinstance(version, str) or "/" not in version:
        raise UnsupportedSchemaVersion(version)
    name, _, rest = version.partition("/")
    major = rest.split(".")[0]
    if name != SCHEMA_NAME or not major.isdigit() or int(major) != SCHEMA_MAJOR:
        raise UnsupportedSchemaVersion(version)


def _require(doc: Mapping[str, Any], key: str, path: str | None = None) -> Any:
    if key not in doc:
        raise ContractViolation(path or key, "field is required")
    return doc[key]


def _enum(enum_cls: type[E], doc: Mapping[str, Any], key: str, path: str) -> E:
    return _enum_value(enum_cls, _require(doc, key, path), path)


def _enum_value(enum_cls: type[E], raw: Any, path: str) -> E:
    try:
        return enum_cls(raw)
    except ValueError:
        raise ContractViolation(
            path, f"{raw!r} is not a known {enum_cls.__name__}; failing closed rather than guessing"
        ) from None


def _check_probability(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1], got {value!r}")


def _choice_to_dict(choice: SizeChoice | None) -> dict[str, Any] | None:
    if choice is None:
        return None
    return {"size_label": choice.size_label, "confidence": choice.confidence}


def _choice_from_dict(raw: Any, path: str) -> SizeChoice | None:
    if raw is None:
        return None
    return SizeChoice(
        size_label=str(_require(raw, "size_label", f"{path}.size_label")),
        confidence=float(_require(raw, "confidence", f"{path}.confidence")),
    )


def _abstain_from_dict(raw: Any) -> AbstainReason | None:
    if raw is None:
        return None
    return AbstainReason(
        code=_enum(AbstainCode, raw, "code", "recommendation.abstain.code"),
        detail_codes=tuple(str(c) for c in raw.get("detail_codes", ())),
    )


def _size_from_dict(raw: Mapping[str, Any], index: int) -> SizeAssessment:
    path = f"sizes[{index}]"
    return SizeAssessment(
        size_label=str(_require(raw, "size_label", f"{path}.size_label")),
        confidence=float(_require(raw, "confidence", f"{path}.confidence")),
        coverage=_enum(Coverage, raw, "coverage", f"{path}.coverage"),
        missing_regions=tuple(
            _enum_value(BodyRegion, r, f"{path}.missing_regions")
            for r in raw.get("missing_regions", ())
        ),
        regions=tuple(
            _delta_from_dict(d, f"{path}.regions[{i}]")
            for i, d in enumerate(_require(raw, "regions", f"{path}.regions"))
        ),
    )


def _delta_from_dict(raw: Mapping[str, Any], path: str) -> RegionDelta:
    ease = _require(raw, "required_ease", f"{path}.required_ease")
    return RegionDelta(
        region=_enum(BodyRegion, raw, "region", f"{path}.region"),
        critical=bool(_require(raw, "critical", f"{path}.critical")),
        delta_cm=float(_require(raw, "delta_cm", f"{path}.delta_cm")),
        delta_sigma_cm=float(_require(raw, "delta_sigma_cm", f"{path}.delta_sigma_cm")),
        stretch_absorbed_cm=float(_require(raw, "stretch_absorbed_cm", f"{path}.stretch_absorbed_cm")),
        required_ease=EaseWindow(
            min_cm=float(_require(ease, "min_cm", f"{path}.required_ease.min_cm")),
            preferred_cm=float(_require(ease, "preferred_cm", f"{path}.required_ease.preferred_cm")),
            max_cm=float(_require(ease, "max_cm", f"{path}.required_ease.max_cm")),
        ),
        classification=_enum(FitClassification, raw, "classification", f"{path}.classification"),
        uncertain=bool(_require(raw, "uncertain", f"{path}.uncertain")),
    )


def _from_choice(choice: SizeChoice) -> list[float]:
    values = [choice.confidence, choice.confidence * 100.0]
    label = _as_number(choice.size_label)
    if label is not None:
        values.append(label)
    return values


def _as_number(label: str) -> float | None:
    try:
        return float(label)
    except ValueError:
        return None
