"""The agent loop.

    question -> [ model decides ] -> tool calls -> results -> [ model interprets ] -> ...
                      ^                                              |
                      +---- re-analyze if a tool failed or a check ---+
                                  disagrees (bounded by max_steps)

The loop is deliberately small and explicit: the model never executes anything itself; it asks
for a tool, we run it (guarded), and the result - including errors - goes back as the next
message. Re-analysis is not a special mode: an `is_error` tool result simply lets the model try
again, and the step budget guarantees termination.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

from .db import Database
from .providers.base import Block, Message, Provider
from .tools import TOOL_SPECS, ToolBox

SYSTEM_PROMPT = """You are a careful data analyst. You answer questions about the database below \
using tools. Never guess numbers: every figure in your answer must come from a tool result.

Workflow: plan briefly, query with run_sql, use run_python for statistics or to double-check a \
number, call make_chart once when a chart helps, then write the final answer.
If a tool returns an error or a result that looks wrong (empty, implausible, inconsistent with \
another result), investigate and re-run a corrected query instead of answering.
The final answer: 2-5 sentences, plain English, state the key numbers and any caveat. \
Do not paste raw tables.

DATABASE SCHEMA ({dialect}):
{schema}
"""


@dataclass
class Step:
    index: int
    tool: str
    input: dict[str, Any]
    output: str
    is_error: bool
    note: str = ""  # any text the model wrote before the call (its short plan)


@dataclass
class AgentResult:
    question: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    sql: list[str] = field(default_factory=list)
    chart_png: bytes | None = None
    stop_reason: str = "end_turn"  # end_turn | max_steps | max_tokens | refusal

    @property
    def chart_base64(self) -> str | None:
        return base64.b64encode(self.chart_png).decode() if self.chart_png else None


def _text_of(blocks: list[Block]) -> str:
    return "\n".join(b["text"] for b in blocks if b.get("type") == "text").strip()


class Agent:
    def __init__(self, provider: Provider, db: Database, max_steps: int = 8):
        self.provider = provider
        self.db = db
        self.max_steps = max_steps
        self.toolbox = ToolBox(db)

    def ask(self, question: str) -> AgentResult:
        system = SYSTEM_PROMPT.format(dialect=self.db.dialect, schema=self.db.schema_description())
        messages: list[Message] = [{"role": "user", "content": question}]
        result = AgentResult(question=question, answer="")

        for _ in range(self.max_steps):
            response = self.provider.complete(system, messages, TOOL_SPECS)
            messages.append({"role": "assistant", "content": response.content})

            calls = [b for b in response.content if b.get("type") == "tool_use"]
            if response.stop_reason != "tool_use" or not calls:
                result.answer = _text_of(response.content)
                result.stop_reason = response.stop_reason
                if not result.answer:
                    result.answer = f"(no answer text; stop reason: {response.stop_reason})"
                return result

            note = _text_of(response.content)
            tool_results: list[Block] = []
            for call in calls:
                out = self.toolbox.run(call["name"], call["input"])
                if call["name"] in ("run_sql", "run_python", "make_chart") and call["input"].get("sql"):
                    result.sql.append(call["input"]["sql"])
                if out.chart_png:
                    result.chart_png = out.chart_png  # last chart wins
                result.steps.append(Step(len(result.steps) + 1, call["name"], call["input"],
                                         out.content, out.is_error, note))
                note = ""
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": out.content,
                    "is_error": out.is_error,
                })
            # All results of one assistant turn go back together in a single user message.
            messages.append({"role": "user", "content": tool_results})

        result.stop_reason = "max_steps"
        result.answer = (
            f"Stopped after {self.max_steps} steps without a final answer. "
            "The trace shows what was tried; try a more specific question."
        )
        return result
