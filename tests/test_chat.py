"""Tests with a mocked client. They never call the real API."""

from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

import chat


class FakeStream:
    text_stream = ("Hel", "lo")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return NS(stop_reason="end_turn", content=["block"])


def client():
    c = MagicMock()
    c.beta.messages.stream.return_value = FakeStream()
    return c


def test_stream_yields_text_and_keeps_final_message():
    reply = chat.ReplyStream(client(), "claude-opus-5-5", "sys", [{"role": "user", "content": "hi"}])
    assert "".join(reply) == "Hello"
    assert reply.final.content == ["block"]


def test_request_caches_and_uses_low_effort():
    c = client()
    list(chat.ReplyStream(c, "claude-opus-5-5", "sys", []))
    kwargs = c.beta.messages.stream.call_args.kwargs
    assert kwargs["cache_control"] == {"type": "ephemeral"}
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["fallbacks"] == "default"
    assert kwargs["betas"] == [chat.FALLBACK_BETA]


def test_haiku_sends_no_fallbacks():
    c = client()
    list(chat.ReplyStream(c, "claude-haiku-5-5", "sys", []))
    kwargs = c.beta.messages.stream.call_args.kwargs
    assert "fallbacks" not in kwargs
    assert "betas" not in kwargs


def test_rejects_model_outside_allowlist():
    with pytest.raises(ValueError):
        chat.ReplyStream(client(), "claude-fable-5-1", "sys", [])


def test_cost_counts_every_token_type():
    usage = NS(
        input_tokens=1_000_000, output_tokens=1_000_000, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    assert chat.turn_cost("claude-opus-5-5", usage) == pytest.approx(24.0)
    usage = NS(
        input_tokens=0, output_tokens=0, cache_creation_input_tokens=1_000_000, cache_read_input_tokens=1_000_000
    )
    assert chat.turn_cost("claude-opus-5-5", usage) == pytest.approx(5.0 + 0.20)


def test_cost_handles_missing_cache_fields():
    usage = NS(input_tokens=1000, output_tokens=0, cache_creation_input_tokens=None, cache_read_input_tokens=None)
    assert chat.turn_cost("claude-haiku-5-5", usage) == pytest.approx(0.0001)


def test_default_budget_fits_one_full_reply():
    assert chat.max_turn_cost(chat.DEFAULT_MODEL) < chat.DEFAULT_BUDGET_USD
