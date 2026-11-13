"""make_chart: SQL result -> PNG, rendered with matplotlib (no model-written plotting code)."""

from __future__ import annotations

import io
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .db import Database

CHART_KINDS = ("bar", "barh", "line", "scatter", "hist")
PALETTE = ["#2f5d8a", "#e0813a", "#4f9d69", "#a45a9c", "#c9514e", "#6b7280"]


class ChartError(ValueError):
    pass


def render_chart(db: Database, spec: dict[str, Any]) -> bytes:
    """Render a chart from ``{kind, sql, x, y, title, xlabel, ylabel}`` and return PNG bytes."""
    kind = spec.get("kind")
    if kind not in CHART_KINDS:
        raise ChartError(f"kind must be one of {CHART_KINDS}")
    result = db.run_select(spec.get("sql", ""), row_limit=2000)
    if not result.rows:
        raise ChartError("The chart query returned no rows.")
    frame = result.to_frame()

    x = spec.get("x")
    y = spec.get("y")
    ys = [y] if isinstance(y, str) else list(y or [])
    needed = ([x] if x else []) + ys
    for col in needed:
        if col not in frame.columns:
            raise ChartError(f"Column '{col}' not in query result {list(frame.columns)}.")
    if kind == "hist":
        if not (x or ys):
            raise ChartError("hist needs x (the column to bin).")
    elif not x or not ys:
        raise ChartError("x and y are required for this chart kind.")

    fig = Figure(figsize=(8, 4.5), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x" if kind == "barh" else "y", color="#e5e7eb")

    if kind == "bar":
        _grouped_bars(ax, frame, x, ys)
        ax.tick_params(axis="x", rotation=30 if len(frame) > 6 else 0)
    elif kind == "barh":
        ax.barh(frame[x].astype(str), frame[ys[0]], color=PALETTE[0])
        ax.invert_yaxis()
    elif kind == "line":
        for i, col in enumerate(ys):
            ax.plot(frame[x], frame[col], color=PALETTE[i % len(PALETTE)], marker="o",
                    markersize=3, label=col)
        ax.tick_params(axis="x", rotation=30 if len(frame) > 6 else 0)
    elif kind == "scatter":
        ax.scatter(frame[x], frame[ys[0]], color=PALETTE[0], alpha=0.7, s=18)
    else:  # hist
        ax.hist(frame[x or ys[0]].dropna(), bins=min(30, max(5, len(frame) // 5)),
                color=PALETTE[0])

    if len(ys) > 1:
        ax.legend(frameon=False)
    ax.set_title(spec.get("title", ""), loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel(spec.get("xlabel") or (x or ""))
    ax.set_ylabel(spec.get("ylabel") or (ys[0] if len(ys) == 1 else ""))
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


def _grouped_bars(ax, frame, x: str, ys: list[str]) -> None:
    width = 0.8 / len(ys)
    positions = list(range(len(frame)))
    for i, col in enumerate(ys):
        offsets = [p - 0.4 + width * (i + 0.5) for p in positions]
        ax.bar(offsets, frame[col], width=width, color=PALETTE[i % len(PALETTE)], label=col)
    ax.set_xticks(positions)
    ax.set_xticklabels(frame[x].astype(str))
