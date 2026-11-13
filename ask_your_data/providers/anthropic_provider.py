"""Claude via the official `anthropic` SDK (tool use, manual agent loop lives in agent.py)."""

from __future__ import annotations

import os

import anthropic

from .base import LLMResponse, Message, ProviderError

DEFAULT_MODEL = "claude-sonnet-5-5"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None, client: anthropic.Anthropic | None = None,
                 max_tokens: int = 4096):
        self.model = model or os.environ.get("ASK_YOUR_DATA_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens
        # Reads ANTHROPIC_API_KEY from the environment when no client is injected.
        self.client = client or anthropic.Anthropic()

    def complete(self, system: str, messages: list[Message], tools: list[dict]) -> LLMResponse:
        try:
            # No temperature/top_p (rejected by this model family), no forced tool_choice:
            # the model decides when to call tools, `thinking` is left at the model default.
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                tools=tools,
                messages=messages,
            )
        except anthropic.AuthenticationError as exc:
            raise ProviderError("Anthropic rejected the API key (check ANTHROPIC_API_KEY).") from exc
        except anthropic.RateLimitError as exc:
            raise ProviderError("Anthropic rate limit hit; retry in a moment.") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError("Could not reach the Anthropic API (network problem).") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc

        # Keep blocks exactly as returned (including thinking blocks) so they can be replayed
        # unchanged on the next turn of the tool loop.
        content = [block.model_dump(exclude_none=True) for block in response.content]
        return LLMResponse(content=content, stop_reason=response.stop_reason or "end_turn")
