"""FitAssessment is the boundary that enforces C1 and C2. These are its teeth."""
import dataclasses
import json
import typing

import pytest

from fitkit.domain.contracts.fit_assessment import (
    SCHEMA_VERSION,
    AbstainCode,
    AbstainReason,
    Coverage,
    FitAssessment,
    FitClassification,
    Recommendation,
    SizeAssessment,
    SizeChoice,
    Verdict,
)
from fitkit.domain.errors import ContractViolation, UnsupportedSchemaVersion

from tests.domain.factories import assessment


class TestSerialisation:
    def test_round_trips_exactly(self):
        original = assessment()
        assert FitAssessment.from_dict(original.to_dict()) == original

    def test_serialisation_is_deterministic(self):
        a, b = assessment().to_dict(), assessment().to_dict()
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    def test_document_is_json_serialisable_without_custom_encoders(self):
        json.dumps(assessment().to_dict())

    def test_stamps_the_schema_version(self):
        assert assessment().to_dict()["schema_version"] == SCHEMA_VERSION

    def test_timestamps_serialise_as_iso_8601_utc(self):
        computed = assessment().to_dict()["inputs_digest"]["computed_at"]
        assert computed.endswith("+00:00")


class TestVersioning:
    def test_rejects_an_incompatible_major_version(self):
        doc = assessment().to_dict()
        doc["schema_version"] = "fit-assessment/2.0.0"
        with pytest.raises(UnsupportedSchemaVersion):
            FitAssessment.from_dict(doc)

    def test_accepts_a_future_minor_version(self):
        """Minors are additive; an old reader must keep working (design 3.3)."""
        doc = assessment().to_dict()
        doc["schema_version"] = "fit-assessment/1.7.0"
        assert FitAssessment.from_dict(doc).schema_version == "fit-assessment/1.7.0"

    def test_ignores_unknown_optional_fields(self):
        doc = assessment().to_dict()
        doc["future_field"] = {"anything": 1}
        doc["sizes"][0]["future_score"] = 0.5
        FitAssessment.from_dict(doc)

    def test_rejects_a_missing_schema_version(self):
        doc = assessment().to_dict()
        del doc["schema_version"]
        with pytest.raises(UnsupportedSchemaVersion):
            FitAssessment.from_dict(doc)

    def test_rejects_a_missing_required_field(self):
        doc = assessment().to_dict()
        del doc["recommendation"]
        with pytest.raises(ContractViolation, match="recommendation"):
            FitAssessment.from_dict(doc)

    def test_unknown_enum_value_fails_closed(self):
        """A renderer that cannot interpret a verdict must not guess."""
        doc = assessment().to_dict()
        doc["recommendation"]["verdict"] = "PROBABLY_FINE"
        with pytest.raises(ContractViolation, match="verdict"):
            FitAssessment.from_dict(doc)

    def test_unknown_classification_fails_closed(self):
        doc = assessment().to_dict()
        doc["sizes"][0]["regions"][0]["classification"] = "SORT_OF_OK"
        with pytest.raises(ContractViolation, match="classification"):
            FitAssessment.from_dict(doc)


class TestVerdictInvariants:
    def _rec(self, **kw) -> Recommendation:
        base = dict(verdict=Verdict.SINGLE, primary=SizeChoice("48", 0.71), alternate=None, abstain=None)
        base.update(kw)
        return Recommendation(**base)

    def test_single_verdict_forbids_an_alternate(self):
        with pytest.raises(ValueError, match="alternate"):
            self._rec(alternate=SizeChoice("50", 0.2))

    def test_two_sizes_requires_an_alternate(self):
        with pytest.raises(ValueError, match="alternate"):
            self._rec(verdict=Verdict.TWO_SIZES)

    def test_abstain_forbids_a_primary(self):
        with pytest.raises(ValueError, match="primary"):
            self._rec(verdict=Verdict.ABSTAIN, abstain=AbstainReason(AbstainCode.NO_SIZE_ACCEPTABLE, ()))

    def test_abstain_requires_a_reason(self):
        with pytest.raises(ValueError, match="abstain"):
            self._rec(verdict=Verdict.ABSTAIN, primary=None)

    def test_a_recommended_size_must_appear_in_the_size_list(self):
        with pytest.raises(ValueError, match="not among the assessed sizes"):
            assessment(recommendation=self._rec(primary=SizeChoice("99", 0.71)))

    @pytest.mark.parametrize("bad", [-0.01, 1.01])
    def test_confidence_must_be_a_probability(self, bad):
        with pytest.raises(ValueError, match="confidence"):
            SizeChoice("48", bad)

    def test_partial_coverage_requires_naming_the_missing_regions(self):
        with pytest.raises(ValueError, match="missing_regions"):
            SizeAssessment(
                size_label="48", confidence=0.5, regions=(), coverage=Coverage.PARTIAL, missing_regions=()
            )

    def test_complete_coverage_forbids_missing_regions(self):
        from fitkit.domain.regions import BodyRegion

        with pytest.raises(ValueError, match="missing_regions"):
            SizeAssessment(
                size_label="48",
                confidence=0.5,
                regions=(),
                coverage=Coverage.COMPLETE,
                missing_regions=(BodyRegion.THIGH,),
            )


def _reachable_dataclasses(cls, seen=None):
    """Every dataclass reachable from the contract root, wherever it is defined."""
    seen = set() if seen is None else seen
    if not dataclasses.is_dataclass(cls) or cls in seen:
        return seen
    seen.add(cls)
    for hint in typing.get_type_hints(cls).values():
        for candidate in (hint, *typing.get_args(hint)):
            for inner in (candidate, *typing.get_args(candidate)):
                _reachable_dataclasses(inner, seen)
    return seen


def _string_fields(cls):
    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        hint = hints[f.name]
        flat = [hint, *typing.get_args(hint)]
        flat += [a for h in flat for a in typing.get_args(h)]
        if any(t is str for t in flat):
            yield f.name


class TestNoFreeText:
    """R1: the engine emits enums and identifiers, never prose."""

    IDENTIFIER_FIELDS = frozenset({
        "assessment_id", "schema_version", "garment_id", "size_system", "size_label",
        "locale", "measurement_backend", "measurement_provenance_id",
        "garment_spec_version", "engine_version", "policy_version",
        "residual_table_version", "detail_codes",
    })

    def test_every_string_field_is_an_identifier_not_prose(self):
        offenders = sorted(
            f"{cls.__name__}.{name}"
            for cls in _reachable_dataclasses(FitAssessment)
            for name in _string_fields(cls)
            if name not in self.IDENTIFIER_FIELDS
        )
        assert offenders == []

    def test_the_contract_reaches_every_type_we_think_it_does(self):
        """Guards the two tests below: a walker that finds nothing passes vacuously."""
        names = {c.__name__ for c in _reachable_dataclasses(FitAssessment)}
        assert {"FitAssessment", "Recommendation", "SizeAssessment", "RegionDelta",
                "EaseWindow", "InputsDigest", "GarmentRef", "FabricSummary"} <= names

    def test_qualitative_judgement_is_carried_by_a_closed_enum(self):
        import enum

        assert issubclass(FitClassification, enum.Enum)
        assert len(FitClassification) >= 5


class TestUncertaintyIsAdjacent:
    """Every measured centimetre in the contract travels with its sigma, or is exempt."""

    EXEMPT = frozenset({
        # Policy constants. Exact by definition -- they are the rule, not a measurement.
        "min_cm", "preferred_cm", "max_cm",
        # Derived from the fabric spec; carries no independent uncertainty in v1.
        # A deliberate gap, recorded rather than hidden (ADR-004).
        "stretch_absorbed_cm",
    })

    def test_every_cm_field_has_a_sigma_sibling_or_a_documented_exemption(self):
        offenders = []
        for cls in _reachable_dataclasses(FitAssessment):
            names = {f.name for f in dataclasses.fields(cls)}
            for name in names:
                if not name.endswith("_cm") or name.endswith("_sigma_cm") or name in self.EXEMPT:
                    continue
                if f"{name[:-3]}_sigma_cm" not in names:
                    offenders.append(f"{cls.__name__}.{name}")
        assert sorted(offenders) == []

    def test_the_exemption_list_is_not_stale(self):
        """An exemption for a field that no longer exists hides a real regression."""
        live = {f.name for cls in _reachable_dataclasses(FitAssessment) for f in dataclasses.fields(cls)}
        assert self.EXEMPT <= live


class TestNumericAllowlist:
    """R4: the Phase 5 guard checks LLM output against exactly this set."""

    def test_contains_every_stated_quantity(self):
        allowed = assessment().numeric_allowlist()
        assert 2.0 in allowed        # |delta_cm| for the waist
        assert 1.4 in allowed        # its sigma
        assert 1.2 in allowed        # stretch absorbed
        assert 48.0 in allowed       # the numeric size label
        assert 71.0 in allowed       # confidence as a percentage
        assert 0.71 in allowed       # confidence as a fraction

    def test_contains_rounded_forms_a_renderer_would_naturally_use(self):
        allowed = assessment().numeric_allowlist()
        assert 1.0 in allowed        # 1.4 -> "about 1 cm"
        assert 6.0 in allowed        # usable extension pct

    def test_excludes_numbers_the_document_never_states(self):
        allowed = assessment().numeric_allowlist()
        assert 3.7 not in allowed
        assert 99.0 not in allowed

    def test_is_derived_from_the_document_alone(self):
        a = assessment()
        assert a.numeric_allowlist() == FitAssessment.from_dict(a.to_dict()).numeric_allowlist()

    def test_permits_reports_membership_within_tolerance(self):
        a = assessment()
        assert a.permits(2.0)
        assert a.permits(2.001)
        assert not a.permits(3.7)


class TestRegionDeltaValidation:
    def _delta(self, **kw):
        from fitkit.domain.policy import EaseWindow
        from fitkit.domain.regions import BodyRegion

        base = dict(
            region=BodyRegion.WAIST, critical=True, delta_cm=-2.0, delta_sigma_cm=1.4,
            stretch_absorbed_cm=1.2,
            required_ease=EaseWindow(min_cm=1.0, preferred_cm=2.0, max_cm=5.0),
            classification=FitClassification.TIGHT, uncertain=False,
        )
        base.update(kw)
        from fitkit.domain.contracts.fit_assessment import RegionDelta

        return RegionDelta(**base)

    def test_a_delta_without_uncertainty_is_rejected(self):
        with pytest.raises(ValueError, match="delta_sigma_cm"):
            self._delta(delta_sigma_cm=0.0)

    def test_a_non_finite_delta_is_rejected(self):
        with pytest.raises(ValueError, match="delta_cm"):
            self._delta(delta_cm=float("inf"))

    def test_negative_stretch_absorption_is_rejected(self):
        with pytest.raises(ValueError, match="stretch_absorbed_cm"):
            self._delta(stretch_absorbed_cm=-0.5)

    def test_negative_usable_extension_is_rejected(self):
        from fitkit.domain.contracts.fit_assessment import FabricSummary
        from fitkit.domain.fabric import RecoveryClass, StretchClass

        with pytest.raises(ValueError, match="usable_extension_pct"):
            FabricSummary(StretchClass.LOW, RecoveryClass.GOOD, -1.0)


class TestMoreVerdictInvariants:
    def test_abstain_forbids_an_alternate(self):
        from fitkit.domain.contracts.fit_assessment import AbstainReason

        with pytest.raises(ValueError, match="alternate"):
            Recommendation(
                verdict=Verdict.ABSTAIN, primary=None, alternate=SizeChoice("50", 0.2),
                abstain=AbstainReason(AbstainCode.NO_SIZE_ACCEPTABLE, ()),
            )

    def test_a_decided_verdict_forbids_an_abstain_reason(self):
        from fitkit.domain.contracts.fit_assessment import AbstainReason

        with pytest.raises(ValueError, match="abstain"):
            Recommendation(
                verdict=Verdict.SINGLE, primary=SizeChoice("48", 0.7), alternate=None,
                abstain=AbstainReason(AbstainCode.NO_SIZE_ACCEPTABLE, ()),
            )

    def test_a_decided_verdict_requires_a_primary(self):
        with pytest.raises(ValueError, match="primary"):
            Recommendation(verdict=Verdict.SINGLE, primary=None, alternate=None, abstain=None)

    def test_an_assessment_must_cover_a_size(self):
        with pytest.raises(ValueError, match="at least one size"):
            assessment(sizes=())

    def test_digest_timestamps_must_be_timezone_aware(self):
        import datetime as dt

        from tests.domain.factories import digest

        with pytest.raises(ValueError, match="timezone"):
            digest(computed_at=dt.datetime(2026, 9, 3, 12, 0, 0))


class TestTwoSizeAndAbstainDocuments:
    def test_a_two_size_document_round_trips(self):
        from tests.domain.factories import assessment as make

        a = make(
            recommendation=Recommendation(
                verdict=Verdict.TWO_SIZES,
                primary=SizeChoice("48", 0.52),
                alternate=SizeChoice("48", 0.41),
                abstain=None,
            )
        )
        assert FitAssessment.from_dict(a.to_dict()) == a
        assert a.recommendation.verdict is Verdict.TWO_SIZES

    def test_an_abstain_document_round_trips_with_its_detail_codes(self):
        from fitkit.domain.contracts.fit_assessment import AbstainReason
        from tests.domain.factories import assessment as make

        a = make(
            recommendation=Recommendation(
                verdict=Verdict.ABSTAIN,
                primary=None,
                alternate=None,
                abstain=AbstainReason(
                    AbstainCode.UNCERTAINTY_EXCEEDS_SIZE_STEP, ("waist_sigma_over_ceiling",)
                ),
            )
        )
        parsed = FitAssessment.from_dict(a.to_dict())
        assert parsed == a
        assert parsed.recommendation.abstain.detail_codes == ("waist_sigma_over_ceiling",)

    def test_a_partial_coverage_document_round_trips(self):
        from fitkit.domain.regions import BodyRegion
        from tests.domain.factories import assessment as make

        a = make(
            sizes=(
                SizeAssessment(
                    size_label="48", confidence=0.6, regions=(),
                    coverage=Coverage.PARTIAL, missing_regions=(BodyRegion.THIGH,),
                ),
            )
        )
        assert FitAssessment.from_dict(a.to_dict()) == a


class TestMalformedDocuments:
    def test_a_structurally_invalid_document_becomes_a_contract_violation(self):
        """A ValueError from an invariant must not escape as a bare ValueError."""
        doc = assessment().to_dict()
        doc["sizes"][0]["coverage"] = "PARTIAL"  # but missing_regions is empty
        with pytest.raises(ContractViolation):
            FitAssessment.from_dict(doc)

    def test_a_non_string_schema_version_is_refused(self):
        doc = assessment().to_dict()
        doc["schema_version"] = 1
        with pytest.raises(UnsupportedSchemaVersion):
            FitAssessment.from_dict(doc)

    def test_a_foreign_schema_name_is_refused(self):
        doc = assessment().to_dict()
        doc["schema_version"] = "something-else/1.0.0"
        with pytest.raises(UnsupportedSchemaVersion):
            FitAssessment.from_dict(doc)

    def test_a_non_numeric_size_label_is_simply_absent_from_the_allowlist(self):
        from tests.domain.factories import assessment as make

        a = make(
            sizes=(SizeAssessment("M", 0.7, (), Coverage.COMPLETE, ()),),
            recommendation=Recommendation(Verdict.SINGLE, SizeChoice("M", 0.7), None, None),
        )
        assert a.permits(0.7)
