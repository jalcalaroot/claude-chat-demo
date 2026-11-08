"""Tests with a mocked client. They never call the real API."""

from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

import chat


class FakeStream:
    def __init__(self, stop_reason="end_turn", content=()):
        self.stop_reason = stop_reason
        self.content = list(content)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        yield NS(type="content_block_start", content_block=NS(type="server_tool_use"))
        yield NS(type="text", text="Hel")
        yield NS(type="text", text="lo")

    def get_final_message(self):
        usage = NS(input_tokens=0, output_tokens=0, server_tool_use=NS(web_search_requests=2))
        return NS(stop_reason=self.stop_reason, content=self.content, usage=usage)


def client():
    c = MagicMock()
    c.beta.messages.stream.return_value = FakeStream()
    return c


def test_stream_yields_text_and_keeps_final_message():
    reply = chat.ReplyStream(client(), "claude-opus-5-5", "sys", [{"role": "user", "content": "hi"}])
    assert "".join(reply) == "Hello"
    assert reply.final.stop_reason == "end_turn"


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


def test_request_declares_web_search_with_a_limit():
    c = client()
    list(chat.ReplyStream(c, "claude-opus-5-5", "sys", []))
    (tool,) = c.beta.messages.stream.call_args.kwargs["tools"]
    assert tool["name"] == "web_search"
    assert tool["max_uses"] == chat.MAX_SEARCHES


def test_search_notifies_and_is_billed():
    called = []
    reply = chat.ReplyStream(client(), "claude-haiku-5-5", "sys", [], on_search=lambda: called.append(1))
    list(reply)
    assert called == [1]
    assert reply.cost() == pytest.approx(2 * chat.SEARCH_PRICE_USD)


def test_pause_turn_resumes_with_the_paused_content_and_no_new_user_message():
    c = client()
    c.beta.messages.stream.side_effect = [FakeStream("pause_turn", ["paused"]), FakeStream("end_turn")]
    reply = chat.ReplyStream(c, "claude-haiku-5-5", "sys", [{"role": "user", "content": "q"}])
    list(reply)
    second = c.beta.messages.stream.call_args_list[1].kwargs["messages"]
    assert second == [{"role": "user", "content": "q"}, {"role": "assistant", "content": ["paused"]}]
    assert [t["role"] for t in reply.assistant_turns] == ["assistant", "assistant"]


def test_resumes_are_limited():
    c = client()
    c.beta.messages.stream.side_effect = lambda **kw: FakeStream("pause_turn")
    list(chat.ReplyStream(c, "claude-haiku-5-5", "sys", []))
    assert c.beta.messages.stream.call_count == chat.MAX_RESUMES + 1


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (["a ![x](http://e/p.png) b"], "a !\\[x](http://e/p.png) b"),
        (["a !", "[x](u)"], "a !\\[x](u)"),
        (["Hello!", " Bye"], "Hello! Bye"),
        (["End!"], "End!"),
    ],
)
def test_images_are_neutralized_even_across_chunks(chunks, expected):
    assert "".join(chat.neutralize_images(chunks)) == expected


def test_sources_are_deduplicated_escaped_and_http_only():
    def cite(url, title):
        return NS(type="web_search_result_location", url=url, title=title)

    block = NS(
        type="text",
        citations=[
            cite("https://a.example/x", "A [b]"),
            cite("https://a.example/x", "dup"),
            cite("javascript:x", "bad"),
        ],
    )
    reply = chat.ReplyStream(client(), "claude-haiku-5-5", "sys", [])
    reply.finals.append(NS(content=[NS(type="server_tool_use"), block, NS(type="text", citations=None)]))
    assert reply.sources() == [("A \\[b\\]", "https://a.example/x")]
