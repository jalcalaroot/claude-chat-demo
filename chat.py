"""Chat logic: allowed models, streaming, web search and cost. No Streamlit code here."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from typing import Any

MAX_TOKENS = 16000
MAX_INPUT_CHARS = 8000
MAX_TURNS = 30
DEFAULT_BUDGET_USD = 1.00

# The default mode picks the fallback model by refusal category.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

MAX_SEARCHES = 3
SEARCH_PRICE_USD = 0.01
MAX_RESUMES = 2  # a long search turn can pause and needs another request

# Direct calls keep max_uses exact and work on every model in the allowlist.
WEB_SEARCH_TOOL = {
    "type": "web_search_20260318",
    "name": "web_search",
    "max_uses": MAX_SEARCHES,
    "allowed_callers": ["direct"],
}


@dataclass(frozen=True)
class Model:
    label: str
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    supports_fallback: bool


# Allowlist: the app can only call these models. Public list prices, USD.
MODELS: dict[str, Model] = {
    "claude-opus-5-5": Model("Claude Opus 5.5", 4.00, 20.00, 0.20, True),
    "claude-sonnet-5-5": Model("Claude Sonnet 5.5", 2.00, 10.00, 0.20, True),
    "claude-haiku-5-5": Model("Claude Haiku 5.5", 0.10, 0.50, 0.01, False),
}
DEFAULT_MODEL = "claude-opus-5-5"
CACHE_WRITE_MULTIPLIER = 1.25


def turn_cost(model_id: str, usage: Any) -> float:
    """Estimate the cost of one API response in USD from the usage it returned."""
    model = MODELS[model_id]
    cache_write = (getattr(usage, "cache_creation_input_tokens", 0) or 0) * CACHE_WRITE_MULTIPLIER
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    tokens = (
        (usage.input_tokens + cache_write) * model.input_per_mtok
        + cache_read * model.cache_read_per_mtok
        + usage.output_tokens * model.output_per_mtok
    ) / 1_000_000
    server_tools = getattr(usage, "server_tool_use", None)
    searches = (getattr(server_tools, "web_search_requests", 0) or 0) if server_tools else 0
    return tokens + searches * SEARCH_PRICE_USD


def max_turn_cost(model_id: str) -> float:
    """Most that the output and searches of one reply can cost."""
    return MAX_TOKENS * MODELS[model_id].output_per_mtok / 1_000_000 + MAX_SEARCHES * SEARCH_PRICE_USD


def neutralize_images(chunks: Iterable[str]) -> Iterator[str]:
    """Stop markdown images from rendering.

    Web content can steer the reply, and an image URL would be fetched by the
    browser as soon as it renders, which could leak conversation text.
    """
    held = ""
    for chunk in chunks:
        text = (held + chunk).replace("![", "!\\[")
        held = "!" if text.endswith("!") else ""
        if held:
            text = text[:-1]
        if text:
            yield text
    if held:
        yield held


def _safe_title(title: str) -> str:
    return re.sub(r"([\[\]\\])", r"\\\1", " ".join(title.split()))


class ReplyStream:
    """Iterate to get text as it arrives; the API responses stay in `finals`."""

    def __init__(
        self, client, model_id: str, system: str, messages: list[dict], on_search: Callable[[], None] | None = None
    ):
        if model_id not in MODELS:
            raise ValueError(f"model not allowed: {model_id}")
        self.client = client
        self.model_id = model_id
        self.system = system
        self.messages = messages
        self.on_search = on_search
        self.finals: list[Any] = []

    def _request(self, messages: list[dict]) -> dict:
        request = {
            "model": self.model_id,
            "max_tokens": MAX_TOKENS,
            # The date lets Claude search for current information with the right year.
            "system": f"{self.system}\n\nToday's date is {date.today().isoformat()}.",
            "messages": messages,
            "tools": [WEB_SEARCH_TOOL],
            "output_config": {"effort": "low"},
            "cache_control": {"type": "ephemeral"},
        }
        if MODELS[self.model_id].supports_fallback:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        return request

    def _text(self, stream) -> Iterator[str]:
        for event in stream:
            if event.type == "text":
                yield event.text
            elif (
                event.type == "content_block_start"
                and event.content_block.type == "server_tool_use"
                and self.on_search is not None
            ):
                self.on_search()

    def __iter__(self):
        messages = list(self.messages)
        for _ in range(MAX_RESUMES + 1):
            with self.client.beta.messages.stream(**self._request(messages)) as stream:
                yield from neutralize_images(self._text(stream))
                final = stream.get_final_message()
            self.finals.append(final)
            if final.stop_reason != "pause_turn":
                return
            # Sent back unchanged, with no extra user message, so the server resumes.
            messages.append({"role": "assistant", "content": final.content})

    @property
    def final(self):
        return self.finals[-1] if self.finals else None

    @property
    def assistant_turns(self) -> list[dict]:
        """Full content blocks: search results and thinking must go back unchanged."""
        return [{"role": "assistant", "content": f.content} for f in self.finals]

    def cost(self) -> float:
        return sum(turn_cost(self.model_id, f.usage) for f in self.finals)

    def sources(self) -> list[tuple[str, str]]:
        """Cited pages as (safe title, url), without duplicates."""
        found: dict[str, str] = {}
        for final in self.finals:
            for block in final.content:
                if block.type != "text":
                    continue
                for citation in block.citations or []:
                    if citation.type == "web_search_result_location" and citation.url.startswith(
                        ("https://", "http://")
                    ):
                        found.setdefault(citation.url, _safe_title(citation.title or citation.url))
        return [(title, url) for url, title in found.items()]
