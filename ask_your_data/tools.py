"""The three tools the agent can call, plus the code that executes them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .charts import CHART_KINDS, ChartError, render_chart
from .db import Database, QueryError
from .sandbox import run_python
from .sql_guard import SQLGuardError

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "run_sql",
        "description": (
            "Run ONE read-only SELECT query against the database and get the rows back as a "
            "table. Writes and DDL are rejected. Results are row-limited, so aggregate in SQL. "
            "If the query errors, read the message and fix the query."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single SELECT statement."}},
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_python",
        "description": (
            "Analyze data with pandas. The `sql` SELECT is executed first (read-only) and its "
            "result is available to your code as the DataFrame `df` (`pd` and `np` are also in "
            "scope). Use print() to output results. No network, no file writes. Use this to "
            "verify a number, compute statistics, or do what SQL cannot."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SELECT producing the data for `df`."},
                "code": {"type": "string", "description": "Python code using `df`; print results."},
            },
            "required": ["sql", "code"],
            "additionalProperties": False,
        },
    },
    {
        "name": "make_chart",
        "description": (
            "Render a PNG chart from a SELECT result. Call it once, when you know which "
            "chart answers the question. `x` and `y` are column names of the query result."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(CHART_KINDS)},
                "sql": {"type": "string", "description": "SELECT producing the chart data."},
                "x": {"type": "string", "description": "Column for the x axis (or the hist column)."},
                "y": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or more numeric columns for the y axis.",
                },
                "title": {"type": "string"},
            },
            "required": ["kind", "sql", "title"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class ToolOutput:
    content: str
    is_error: bool = False
    chart_png: bytes | None = None


class ToolBox:
    """Executes tool calls. Errors become results the model can read and react to."""

    def __init__(self, db: Database, python_timeout_s: float = 10.0):
        self.db = db
        self.python_timeout_s = python_timeout_s

    def run(self, name: str, tool_input: dict[str, Any]) -> ToolOutput:
        try:
            if name == "run_sql":
                return ToolOutput(self.db.run_select(tool_input["sql"]).to_text())
            if name == "run_python":
                frame = self.db.run_select(tool_input["sql"], row_limit=100_000).to_frame()
                result = run_python(tool_input["code"], frame, self.python_timeout_s)
                return ToolOutput(result.to_text(), is_error=not result.ok)
            if name == "make_chart":
                png = render_chart(self.db, tool_input)
                return ToolOutput("Chart rendered and attached to the answer.", chart_png=png)
            return ToolOutput(f"ERROR: unknown tool '{name}'.", is_error=True)
        except (SQLGuardError, QueryError, ChartError) as exc:
            return ToolOutput(f"ERROR: {exc}", is_error=True)
        except KeyError as exc:
            return ToolOutput(f"ERROR: missing required argument {exc}.", is_error=True)
