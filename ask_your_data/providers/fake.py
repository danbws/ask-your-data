"""Deterministic scripted provider. No network, no API key.

Used by the tests, by CI and by `--demo`. It does NOT understand questions: it replays a fixed
script of tool calls. Script items are either a ready LLMResponse or a callable that receives
the conversation so far (so the final answer can quote real tool output).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

from .base import Block, LLMResponse, Message, ProviderError

ScriptItem = Union[LLMResponse, Callable[[list[Message]], LLMResponse]]  # noqa: UP007

DEMO_QUESTION = "Which machines have the highest re-dye rate, and does it add to lead time?"


def tool_call(name: str, tool_input: dict, call_id: str, text: str = "") -> LLMResponse:
    content: list[Block] = []
    if text:
        content.append({"type": "text", "text": text})
    content.append({"type": "tool_use", "id": call_id, "name": name, "input": tool_input})
    return LLMResponse(content, "tool_use")


def final_text(text: str) -> LLMResponse:
    return LLMResponse([{"type": "text", "text": text}], "end_turn")


def tool_results(messages: list[Message]) -> list[str]:
    """All tool_result texts in the conversation, in order."""
    out = []
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg["content"], list):
            out += [b["content"] for b in msg["content"] if b.get("type") == "tool_result"]
    return out


class FakeProvider:
    name = "fake"

    def __init__(self, script: list[ScriptItem]):
        self.script = list(script)
        self.calls = 0

    def complete(self, system: str, messages: list[Message], tools: list[dict]) -> LLMResponse:
        if self.calls >= len(self.script):
            raise ProviderError("FakeProvider script exhausted.")
        item = self.script[self.calls]
        self.calls += 1
        return item(messages) if callable(item) else item


def demo_provider() -> FakeProvider:
    """Scripted analysis for the sample textile dataset, including one deliberate mistake that
    the agent has to recover from (to show the re-analysis path)."""
    redye_sql = (
        "SELECT machine_id, COUNT(*) AS batches, SUM(redye) AS redyes, "
        "ROUND(100.0 * SUM(redye) / COUNT(*), 1) AS redye_rate_pct "
        "FROM dye_batches GROUP BY machine_id ORDER BY redye_rate_pct DESC"
    )
    lead_sql = (
        "SELECT SUM(redye) AS redyes_in_order, COUNT(*) AS batches, "
        "o.lead_time_days AS lead_time_days "
        "FROM dye_batches b JOIN production_orders o ON o.order_id = b.order_id "
        "WHERE o.status = 'completed' GROUP BY o.order_id"
    )
    lead_code = (
        "g = df.groupby('redyes_in_order')['lead_time_days'].agg(['count', 'mean']).round(2)\n"
        "print(g.to_string())\n"
        "base = g.loc[0, 'mean']\n"
        "print('lead time without any re-dye:', base, 'days')\n"
        "print('correlation redyes vs lead time:', round(df['redyes_in_order']"
        ".corr(df['lead_time_days']), 3))\n"
    )

    def answer(messages: list[Message]) -> LLMResponse:
        results = tool_results(messages)
        table = next(r for r in results if "redye_rate_pct" in r and not r.startswith("ERROR"))
        rows = [ln for ln in table.splitlines() if ln.startswith("| ") and "---" not in ln][1:]
        top = [c.strip() for c in rows[0].strip("|").split("|")]
        second = [c.strip() for c in rows[1].strip("|").split("|")]
        stats = next(r for r in results if "lead time without any re-dye" in r)
        base_line = next(ln for ln in stats.splitlines() if ln.startswith("lead time without"))
        corr_line = next(ln for ln in stats.splitlines() if ln.startswith("correlation"))
        corr = float(corr_line.rsplit(":", 1)[1])
        effect = "Re-dyes do add lead time" if corr > 0.1 else "Re-dyes show little lead-time effect"
        return final_text(
            f"{top[0]} has the highest re-dye rate: {top[3]}% of its {top[1]} batches were "
            f"re-dyed, versus {second[3]}% for the next machine ({second[0]}). "
            f"{effect}. Per the analysis: {base_line}; {corr_line}. "
            "See the chart for all machines."
        )

    return FakeProvider(
        [
            tool_call(
                "run_sql",
                {"sql": "SELECT machine_id, AVG(redo_flag) FROM dye_batches GROUP BY machine_id"},
                "t1",
                "Plan: re-dye rate per machine, then check the effect on order lead time.",
            ),
            tool_call(
                "run_sql", {"sql": redye_sql}, "t2",
                "That column does not exist; the flag is called `redye`. Re-running.",
            ),
            tool_call(
                "run_python", {"sql": lead_sql, "code": lead_code}, "t3",
                "Cross-checking the lead-time effect with pandas.",
            ),
            tool_call(
                "make_chart",
                {
                    "kind": "bar",
                    "sql": "SELECT machine_id, ROUND(100.0 * SUM(redye) / COUNT(*), 1) "
                           "AS redye_rate_pct FROM dye_batches GROUP BY machine_id "
                           "ORDER BY redye_rate_pct DESC",
                    "x": "machine_id",
                    "y": ["redye_rate_pct"],
                    "title": "Re-dye rate by machine (% of batches)",
                },
                "t4",
            ),
            answer,
        ]
    )
