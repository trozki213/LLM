"""Explanation layer: the template is the reference, the guards are the teeth."""
import pytest

from fitkit.domain.contracts.explanation import Explanation
from fitkit.domain.contracts.fit_assessment import (
    AbstainCode,
    AbstainReason,
    Coverage,
    FitClassification,
    Recommendation,
    SizeAssessment,
    SizeChoice,
    Verdict,
)
from fitkit.domain.errors import BackendUnavailable
from fitkit.explanation import (
    AbstentionGuard,
    BannedClaimGuard,
    GuardedRenderer,
    LengthGuard,
    LlmRenderer,
    NumericGuard,
    TemplateRenderer,
    build_prompt,
    default_guards,
)

from tests.domain.factories import assessment
from tests.fakes import ScriptedLlmClient


def en(**overrides):
    """The shared fixture renders in Italian; these tests read the English output."""
    from fitkit.domain.contracts.fit_assessment import RenderHints
    from fitkit.domain.policy import Tone

    overrides.setdefault("render_hints", RenderHints(locale="en-GB", tone=Tone.NEUTRAL))
    return assessment(**overrides)


def abstaining(**kw):
    return en(
        recommendation=Recommendation(
            verdict=Verdict.ABSTAIN,
            primary=None,
            alternate=None,
            abstain=AbstainReason(AbstainCode.UNCERTAINTY_EXCEEDS_SIZE_STEP, ("waist",)),
        ),
        **kw,
    )


class TestTemplateRenderer:
    def test_renders_the_recommended_size(self):
        text = TemplateRenderer().render(en()).text
        assert "48" in text
        assert "waist" in text.lower()

    def test_output_is_exactly_reproducible(self):
        a = en()
        assert TemplateRenderer().render(a).text == TemplateRenderer().render(a).text

    def test_reads_the_contract_rather_than_reciting_boilerplate(self):
        """Change a number in the document and the prose must change with it."""
        base = TemplateRenderer().render(en()).text
        moved = en(
            sizes=(
                SizeAssessment(
                    size_label="48",
                    confidence=0.71,
                    regions=(
                        _delta(delta_cm=-6.0, classification=FitClassification.MUCH_TOO_TIGHT),
                    ),
                    coverage=Coverage.COMPLETE,
                    missing_regions=(),
                ),
            )
        )
        assert TemplateRenderer().render(moved).text != base
        assert "6" in TemplateRenderer().render(moved).text

    def test_renders_italian(self):
        from fitkit.domain.contracts.fit_assessment import RenderHints
        from fitkit.domain.policy import Tone

        a = en(render_hints=RenderHints(locale="it-IT", tone=Tone.NEUTRAL))
        text = TemplateRenderer().render(a).text
        assert "Ordina" in text
        assert "vita" in text.lower()

    def test_an_unsupported_locale_falls_back_rather_than_failing(self):
        from fitkit.domain.contracts.fit_assessment import RenderHints
        from fitkit.domain.policy import Tone

        a = en(render_hints=RenderHints(locale="ja-JP", tone=Tone.NEUTRAL))
        assert "Order the 48" in TemplateRenderer().render(a).text

    def test_abstains_without_naming_a_size(self):
        text = TemplateRenderer().render(abstaining()).text
        assert "48" not in text
        assert "confidently" in text

    def test_reports_two_sizes_when_the_engine_did(self):
        a = en(
            recommendation=Recommendation(
                Verdict.TWO_SIZES, SizeChoice("48", 0.52), SizeChoice("48", 0.33), None
            )
        )
        assert "Either" in TemplateRenderer().render(a).text

    def test_mentions_the_fabric_absorbing_a_shortfall(self):
        assert "fabric takes up" in TemplateRenderer().render(en()).text

    def test_declares_partial_coverage(self):
        from fitkit.domain.regions import BodyRegion

        a = en(
            sizes=(
                SizeAssessment("48", 0.71, (_delta(),), Coverage.PARTIAL, (BodyRegion.THIGH,)),
            )
        )
        assert "thigh" in TemplateRenderer().render(a).text

    def test_flags_an_uncertain_region(self):
        a = en(
            sizes=(SizeAssessment("48", 0.71, (_delta(uncertain=True),), Coverage.COMPLETE, ()),)
        )
        assert "approximate" in TemplateRenderer().render(a).text

    def test_the_template_passes_its_own_guards(self):
        """The reference renderer must satisfy the rules the LLM is held to."""
        for a in (en(), abstaining()):
            text = TemplateRenderer().render(a).text
            for guard in default_guards():
                assert guard.check(a, text) == (), (guard.guard_id, text)


class TestNumericGuard:
    def test_accepts_numbers_the_document_states(self):
        a = en()
        assert NumericGuard().check(a, "The waist is 2 cm tight, give or take 1.4 cm.") == ()

    def test_rejects_an_invented_number(self):
        v = NumericGuard().check(en(), "You have about 3.7 cm of room.")
        assert v and v[0].code == "unsupported_number"

    def test_accepts_a_decimal_comma(self):
        assert NumericGuard().check(en(), "circa 1,4 cm") == ()

    def test_rejects_a_plausible_but_unstated_conversion(self):
        """Inches are exactly the kind of arithmetic the LLM must not do."""
        v = NumericGuard().check(en(), "That is 0.79 inches.")
        assert v


class TestBannedClaimGuard:
    def test_rejects_reassurance_about_a_tight_fit(self):
        v = BannedClaimGuard().check(en(), "The waist is -2 cm but it will feel roomy.")
        assert v and v[0].code == "reassurance_over_tight_fit"

    def test_allows_accurate_language_about_a_tight_fit(self):
        assert BannedClaimGuard().check(en(), "The waist sits tighter than intended.") == ()

    def test_rejects_constriction_language_about_a_loose_fit(self):
        a = en(
            sizes=(
                SizeAssessment(
                    "48", 0.71,
                    (_delta(delta_cm=9.0, classification=FitClassification.MUCH_TOO_LOOSE),),
                    Coverage.COMPLETE, (),
                ),
            )
        )
        v = BannedClaimGuard().check(a, "It will feel snug.")
        assert v and v[0].code == "constriction_over_loose_fit"

    def test_only_the_described_size_constrains_the_language(self):
        """A tight size we are not recommending must not gag the text about the one we are."""
        a = en(
            sizes=(
                SizeAssessment("48", 0.71, (_delta(delta_cm=4.0, classification=FitClassification.RELAXED),), Coverage.COMPLETE, ()),
                SizeAssessment("46", 0.29, (_delta(delta_cm=-9.0, classification=FitClassification.MUCH_TOO_TIGHT),), Coverage.COMPLETE, ()),
            )
        )
        assert BannedClaimGuard().check(a, "The 48 is relaxed at the waist.") == ()


class TestAbstentionGuard:
    def test_rejects_naming_a_size_while_abstaining(self):
        v = AbstentionGuard().check(abstaining(), "Honestly, just get the 48.")
        assert v and v[0].code == "size_named_while_abstaining"

    def test_allows_prose_that_names_no_size(self):
        assert AbstentionGuard().check(abstaining(), "We cannot call this one.") == ()

    def test_is_inactive_when_the_engine_did_decide(self):
        assert AbstentionGuard().check(en(), "Order the 48.") == ()


class TestGuardedRenderer:
    ADVERSARIAL = [
        "You have 3.7 cm of room at the waist.",          # invented number
        "The waist will feel roomy.",                      # contradicts TIGHT
        "Order the 52 instead.",                           # a size that does not exist
        "Waist is 2 cm tight. " * 200,                     # too long
        "Approximately 8.3 cm of ease at the hip.",        # unstated number
    ]

    @pytest.mark.parametrize("bad", ADVERSARIAL)
    def test_every_adversarial_completion_is_caught_and_replaced(self, bad):
        a = en()
        renderer = GuardedRenderer(LlmRenderer(ScriptedLlmClient(bad)))
        result = renderer.render(a)
        assert result.degraded is True
        assert result.text == TemplateRenderer().render(a).text
        assert any("numeric" in n or "banned_claim" in n or "length" in n for n in result.notes)

    def test_a_clean_completion_passes_through(self):
        good = "Order the 48. The waist sits about 2 cm tighter than intended."
        result = GuardedRenderer(LlmRenderer(ScriptedLlmClient(good))).render(en())
        assert result.degraded is False
        assert result.text == good
        assert "guards_passed" in result.notes

    def test_an_unavailable_llm_degrades_rather_than_failing(self):
        class Dead:
            renderer_id = "dead/1"

            def render(self, assessment):
                raise BackendUnavailable("down")

        result = GuardedRenderer(Dead()).render(en())
        assert result.degraded is True
        assert "BackendUnavailable" in result.notes

    def test_abstention_is_enforced_on_the_llm(self):
        renderer = GuardedRenderer(LlmRenderer(ScriptedLlmClient("Get the 48, it will be fine.")))
        result = renderer.render(abstaining())
        assert result.degraded is True
        assert "48" not in result.text


class TestPrompt:
    def test_carries_the_whole_document(self):
        prompt = build_prompt(en())
        assert "fit-assessment/1.0.0" in prompt
        assert "TIGHT" in prompt

    def test_states_the_rules_that_the_guards_enforce(self):
        prompt = build_prompt(en())
        assert "must already appear in the JSON" in prompt
        assert "Do not choose or suggest a size" in prompt

    def test_asks_for_the_document_language(self):
        from fitkit.domain.contracts.fit_assessment import RenderHints
        from fitkit.domain.policy import Tone

        a = en(render_hints=RenderHints(locale="it-IT", tone=Tone.NEUTRAL))
        assert "Italian" in build_prompt(a)


class TestC2SystemRunsWithoutAnLlm:
    def test_the_llm_adapter_is_imported_by_nothing_else(self):
        import ast
        import pathlib

        root = pathlib.Path("src/fitkit")
        offenders = []
        for path in root.rglob("*.py"):
            if path.name == "anthropic_client.py":
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and "anthropic_client" in (node.module or ""):
                    offenders.append(str(path))
        assert offenders == []

    def test_the_package_never_imports_the_vendor_sdk_at_module_scope(self):
        import sys

        assert "anthropic" not in sys.modules


def _delta(**overrides):
    from fitkit.domain.contracts.fit_assessment import RegionDelta
    from fitkit.domain.policy import EaseWindow
    from fitkit.domain.regions import BodyRegion

    fields = dict(
        region=BodyRegion.WAIST,
        critical=True,
        delta_cm=-2.0,
        delta_sigma_cm=1.4,
        stretch_absorbed_cm=1.2,
        required_ease=EaseWindow(1.0, 2.0, 5.0),
        classification=FitClassification.TIGHT,
        uncertain=False,
    )
    fields.update(overrides)
    return RegionDelta(**fields)
