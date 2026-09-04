"""The Anthropic adapter. The only module in the project that knows an LLM vendor exists.

`anthropic` is an optional extra: importing this module without it raises, and nothing
else in the package imports this module, so the system builds and tests without it.
"""

from __future__ import annotations

from fitkit.domain.errors import BackendTimeout, BackendUnavailable

DEFAULT_MODEL = "claude-opus-5"


class AnthropicLlmClient:
    """Note on determinism: temperature and the other sampling parameters were removed
    on the current Claude models and are rejected, so identical output cannot be
    requested. Reproducibility comes from the template renderer and the guards."""

    def __init__(self, *, model: str = DEFAULT_MODEL, client=None) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - exercised by env, not tests
                raise BackendUnavailable(
                    "the anthropic package is not installed; install fitkit[llm] or inject "
                    "a client. The system runs without it on the template renderer."
                ) from exc
            client = anthropic.Anthropic()
        self._client = client
        self._model = model

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # translated at the boundary; no foreign type escapes
            if "timeout" in type(exc).__name__.lower():
                raise BackendTimeout(str(exc)) from exc
            raise BackendUnavailable(str(exc)) from exc

        if getattr(response, "stop_reason", None) == "refusal":
            raise BackendUnavailable("the model declined to answer; falling back to the template")
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
