"""Chat logic: allowed models, streaming and cost. No Streamlit code here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_TOKENS = 16000
MAX_INPUT_CHARS = 8000
MAX_TURNS = 30
DEFAULT_BUDGET_USD = 0.50

# The default mode picks the fallback model by refusal category.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


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
    """Estimate the cost of one reply in USD from the usage the API returned."""
    model = MODELS[model_id]
    cache_write = (getattr(usage, "cache_creation_input_tokens", 0) or 0) * CACHE_WRITE_MULTIPLIER
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    return (
        (usage.input_tokens + cache_write) * model.input_per_mtok
        + cache_read * model.cache_read_per_mtok
        + usage.output_tokens * model.output_per_mtok
    ) / 1_000_000


def max_turn_cost(model_id: str) -> float:
    """Most that the output of one reply can cost."""
    return MAX_TOKENS * MODELS[model_id].output_per_mtok / 1_000_000


class ReplyStream:
    """Iterate to get text as it arrives; `final` holds the full message afterwards."""

    def __init__(self, client, model_id: str, system: str, messages: list[dict]):
        if model_id not in MODELS:
            raise ValueError(f"model not allowed: {model_id}")
        self.client = client
        self.model_id = model_id
        self.system = system
        self.messages = messages
        self.final = None

    def _request(self) -> dict:
        request = {
            "model": self.model_id,
            "max_tokens": MAX_TOKENS,
            "system": self.system,
            "messages": self.messages,
            "output_config": {"effort": "low"},
            "cache_control": {"type": "ephemeral"},
        }
        if MODELS[self.model_id].supports_fallback:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        return request

    def __iter__(self):
        with self.client.beta.messages.stream(**self._request()) as stream:
            yield from stream.text_stream
            self.final = stream.get_final_message()
