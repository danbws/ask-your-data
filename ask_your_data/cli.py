"""Command line: `ask-your-data --demo`, `ask-your-data "question" --csv file.csv`, `--serve`."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from .agent import Agent, AgentResult
from .db import SQLiteDatabase, ingest_csvs, open_database
from .providers import AnthropicProvider, ProviderError, demo_provider
from .providers.fake import DEMO_QUESTION
from .sample_data import build_sample_db


def _print_result(result: AgentResult, chart_out: Path | None) -> None:
    print(f"\nQUESTION: {result.question}\n")
    for step in result.steps:
        if step.note:
            print(f"[agent] {step.note}")
        status = "ERROR" if step.is_error else "ok"
        print(f"--- step {step.index}: {step.tool} ({status})")
        for key, value in step.input.items():
            print(f"    {key}: {value}")
        first_lines = "\n".join(step.output.splitlines()[:8])
        print("    -> " + first_lines.replace("\n", "\n       "))
    print(f"\nANSWER ({result.stop_reason}):\n{result.answer}")
    if result.chart_png and chart_out:
        chart_out.parent.mkdir(parents=True, exist_ok=True)
        chart_out.write_bytes(result.chart_png)
        print(f"\nChart saved to {chart_out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ask-your-data", description=__doc__)
    parser.add_argument("question", nargs="?", help="question in plain English")
    parser.add_argument("--demo", action="store_true",
                        help="run the scripted demo on the sample data (no API key needed)")
    parser.add_argument("--csv", action="append", default=[], metavar="FILE",
                        help="CSV file to load (repeatable; one table each)")
    parser.add_argument("--db", metavar="URL", help="sqlite:///file.db or postgresql://...")
    parser.add_argument("--chart-out", type=Path, default=Path("out/chart.png"))
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--serve", action="store_true", help="start the API + web UI")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.serve:
        import uvicorn

        uvicorn.run("ask_your_data.api:app_factory", factory=True, host="127.0.0.1",
                    port=args.port)
        return 0

    workdir = Path(tempfile.mkdtemp(prefix="ask_your_data_"))
    if args.demo:
        db = SQLiteDatabase(build_sample_db(workdir / "sample.db"))
        provider = demo_provider()
        question = args.question or DEMO_QUESTION
        print("[demo mode] scripted FakeProvider on synthetic data - no LLM is called.")
        if args.question:
            print("[demo mode] note: the script ignores your question text.")
    else:
        if not args.question:
            parser.error("give a question, or use --demo / --serve")
        if args.db:
            db = open_database(args.db)
        elif args.csv:
            db_path = workdir / "data.db"
            ingest_csvs({Path(c).name: c for c in args.csv}, db_path)
            db = SQLiteDatabase(db_path)
        else:
            parser.error("give --csv FILE or --db URL")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY is not set. Set it, or try: ask-your-data --demo",
                  file=sys.stderr)
            return 2
        provider = AnthropicProvider()
        question = args.question

    try:
        result = Agent(provider, db, max_steps=args.max_steps).ask(question)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_result(result, args.chart_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
