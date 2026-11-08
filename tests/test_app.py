"""Runs the whole Streamlit script with a mocked client."""

from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import anthropic
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


class FakeStream:
    def __init__(self, stop_reason):
        self.stop_reason = stop_reason

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        yield NS(type="text", text="Hi")
        yield NS(type="text", text=" there")

    def get_final_message(self):
        usage = NS(input_tokens=10, output_tokens=10, cache_creation_input_tokens=0, cache_read_input_tokens=0)
        return NS(stop_reason=self.stop_reason, content=[], usage=usage, model="claude-served-model")


@pytest.fixture
def run_turn(monkeypatch):
    def _run(stop_reason):
        st.cache_resource.clear()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        client = MagicMock()
        client.beta.messages.stream.return_value = FakeStream(stop_reason)
        monkeypatch.setattr(anthropic, "Anthropic", lambda: client)
        app = AppTest.from_file("../app.py", default_timeout=20).run()
        app.chat_input[0].set_value("hello").run()
        return app

    return _run


def test_normal_turn_is_kept(run_turn):
    app = run_turn("end_turn")
    assert not app.exception
    assert [m["role"] for m in app.session_state.display] == ["user", "assistant"]
    assert app.session_state.display[1]["content"] == "Hi there"
    assert any("Answered by claude-served-model" in c.value for c in app.caption)
    assert app.session_state.pending is None


def test_refusal_is_reported_and_not_kept(run_turn):
    app = run_turn("refusal")
    assert not app.exception
    assert app.session_state.history == []
    assert app.session_state.pending is None
    assert any("declined" in w.value for w in app.warning)
    assert app.session_state.spent > 0


def test_max_tokens_keeps_the_turn_and_warns(run_turn):
    app = run_turn("max_tokens")
    assert len(app.session_state.history) == 2
    assert any("length limit" in w.value for w in app.warning)
