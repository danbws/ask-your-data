"""Provider interface: the only thing the agent knows about an LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

# Messages and content blocks use the Anthropic Messages shape as plain dicts:
#   {"role": "user" | "assistant", "content": str | list[block]}
#   block: {"type": "text", "text": ...} | {"type": "tool_use", "id", "name", "input"}
#        | {"type": "tool_result", "tool_use_id", "content", "is_error"}
Message = dict[str, Any]
Block = dict[str, Any]


class ProviderError(RuntimeError):
    """The LLM provider failed (auth, rate limit, network...). Message is user-presentable."""


@dataclass
class LLMResponse:
    content: list[Block]
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens" | "refusal" | ...


class Provider(Protocol):
    name: str

    def complete(self, system: str, messages: list[Message], tools: list[dict]) -> LLMResponse: ...
