"""Renderer strategies and the guard decorator.

Strategy varies which renderer produces the prose (template, LLM, an experiment arm).
Decorator varies which guards are applied, and keeps each one independently testable and
reusable across every renderer -- including ones added later, which is exactly how C1
would otherwise erode.
"""

from __future__ import annotations

from fitkit.domain.contracts.explanation import Explanation
from fitkit.domain.contracts.fit_assessment import FitAssessment
from fitkit.domain.errors import BackendTimeout, BackendUnavailable
from fitkit.domain.ports import ExplanationRenderer, LlmClient
from fitkit.explanation.guards import Guard, default_guards
from fitkit.explanation.prompt import build_prompt
from fitkit.explanation.template import TemplateRenderer

DEFAULT_MAX_TOKENS = 400


class LlmRenderer:
    """Adapter over the LLM port. It phrases; it never computes."""

    renderer_id = "llm/1"

    def __init__(self, client: LlmClient, *, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def render(self, assessment: FitAssessment) -> Explanation:
        text = self._client.complete(build_prompt(assessment), max_tokens=self._max_tokens)
        return Explanation(
            text=text.strip() or "(empty)", renderer_id=self.renderer_id, degraded=False
        )


class GuardedRenderer:
    """Runs a renderer, checks what it said, and falls back to the template if it lied.

    Reproducibility comes from here and from the template, never from asking the model
    to be deterministic: sampling parameters were removed from the current Claude models,
    so there is no `temperature=0` to hide behind.
    """

    def __init__(
        self,
        inner: ExplanationRenderer,
        *,
        fallback: ExplanationRenderer | None = None,
        guards: tuple[Guard, ...] | None = None,
    ) -> None:
        self._inner = inner
        self._fallback = fallback or TemplateRenderer()
        self._guards = guards if guards is not None else default_guards()

    @property
    def renderer_id(self) -> str:
        return f"guarded({self._inner.renderer_id})"

    def render(self, assessment: FitAssessment) -> Explanation:
        try:
            candidate = self._inner.render(assessment)
        except (BackendUnavailable, BackendTimeout) as exc:
            return self._degrade(assessment, (f"{type(exc).__name__}",))

        violations = tuple(
            v for guard in self._guards for v in guard.check(assessment, candidate.text)
        )
        if violations:
            return self._degrade(
                assessment, tuple(f"{v.guard_id}:{v.code}:{v.detail}" for v in violations)
            )
        return Explanation(
            text=candidate.text,
            renderer_id=candidate.renderer_id,
            degraded=False,
            notes=candidate.notes + ("guards_passed",),
        )

    def _degrade(self, assessment: FitAssessment, notes: tuple[str, ...]) -> Explanation:
        safe = self._fallback.render(assessment)
        return Explanation(
            text=safe.text,
            renderer_id=safe.renderer_id,
            degraded=True,
            notes=safe.notes + notes,
        )
