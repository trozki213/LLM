"""Explanation is the one domain type that legitimately carries prose."""
import pytest

from fitkit.domain.contracts.explanation import Explanation


def explanation(**overrides) -> Explanation:
    fields = dict(text="The 48 should sit close at the waist.", renderer_id="template/1", degraded=False)
    fields.update(overrides)
    return Explanation(**fields)


class TestExplanation:
    def test_carries_text_and_its_producer(self):
        e = explanation()
        assert e.renderer_id == "template/1"
        assert e.degraded is False

    def test_rejects_empty_text(self):
        with pytest.raises(ValueError, match="must not be empty"):
            explanation(text="   ")

    def test_rejects_an_anonymous_renderer(self):
        """Without renderer_id we cannot measure the LLM against the template (design 7.3)."""
        with pytest.raises(ValueError, match="renderer_id"):
            explanation(renderer_id="")

    def test_notes_are_coerced_to_an_immutable_tuple(self):
        notes = ["guard:numeric_ok"]
        e = explanation(notes=notes)
        notes.append("mutated")
        assert e.notes == ("guard:numeric_ok",)

    def test_degradation_is_explicit_not_inferred(self):
        assert explanation(degraded=True, renderer_id="template/1").degraded is True
