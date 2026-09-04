"""The vendor adapter. Exercised through an injected client, so no SDK and no network."""
import pytest

from fitkit.domain.errors import BackendTimeout, BackendUnavailable
from fitkit.explanation.anthropic_client import DEFAULT_MODEL, AnthropicLlmClient


class _Block:
    def __init__(self, text, type="text"):
        self.text, self.type = text, type


class _Response:
    def __init__(self, blocks, stop_reason="end_turn"):
        self.content, self.stop_reason = blocks, stop_reason


class _Messages:
    def __init__(self, response=None, error=None):
        self._response, self._error = response, error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._response


class _Client:
    def __init__(self, response=None, error=None):
        self.messages = _Messages(response, error)


class _VendorTimeout(Exception):
    """Stands in for the SDK's APITimeoutError, matched by name at the boundary."""


class TestAnthropicAdapter:
    def test_returns_the_text_blocks(self):
        client = _Client(_Response([_Block("Order the 48."), _Block("", "thinking")]))
        assert AnthropicLlmClient(client=client).complete("p", max_tokens=100) == "Order the 48."

    def test_defaults_to_the_current_model(self):
        client = _Client(_Response([_Block("ok")]))
        AnthropicLlmClient(client=client).complete("p", max_tokens=100)
        assert client.messages.calls[0]["model"] == DEFAULT_MODEL
        assert DEFAULT_MODEL == "claude-opus-5"

    def test_sends_no_sampling_parameters(self):
        """They were removed on the current models and are rejected; determinism comes
        from the template renderer and the guards instead."""
        client = _Client(_Response([_Block("ok")]))
        AnthropicLlmClient(client=client).complete("p", max_tokens=100)
        sent = client.messages.calls[0]
        assert not {"temperature", "top_p", "top_k"} & set(sent)

    def test_a_refusal_degrades_rather_than_returning_prose(self):
        client = _Client(_Response([_Block("I can't help")], stop_reason="refusal"))
        with pytest.raises(BackendUnavailable, match="declined"):
            AnthropicLlmClient(client=client).complete("p", max_tokens=100)

    def test_a_timeout_is_translated(self):
        with pytest.raises(BackendTimeout):
            AnthropicLlmClient(client=_Client(error=_VendorTimeout("slow"))).complete(
                "p", max_tokens=100
            )

    def test_any_other_failure_becomes_backend_unavailable(self):
        with pytest.raises(BackendUnavailable):
            AnthropicLlmClient(client=_Client(error=RuntimeError("boom"))).complete(
                "p", max_tokens=100
            )

    def test_no_foreign_exception_escapes(self):
        from fitkit.domain.errors import FitKitError

        for error in (RuntimeError(), ValueError(), _VendorTimeout()):
            with pytest.raises(FitKitError):
                AnthropicLlmClient(client=_Client(error=error)).complete("p", max_tokens=1)

    def test_it_satisfies_the_llm_port(self):
        from fitkit.domain import ports
        from tests.domain.test_ports import assert_conforms

        assert_conforms(AnthropicLlmClient(client=_Client()), ports.LlmClient)
