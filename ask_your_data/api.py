"""FastAPI backend: upload a CSV (or connect a DB), ask a question, get answer + trace + chart."""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .agent import Agent
from .db import Database, SQLiteDatabase, ingest_csvs, open_database, sanitize_name
from .providers import AnthropicProvider, Provider, ProviderError, demo_provider
from .sample_data import build_sample_db

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
STATIC = Path(__file__).parent / "static"


@dataclass
class Dataset:
    id: str
    name: str
    db: Database
    tables: list[str]
    is_sample: bool = False


class ConnectRequest(BaseModel):
    url: str = Field(description="sqlite:///path/to.db or postgresql://...")


class AskRequest(BaseModel):
    dataset_id: str
    question: str = Field(min_length=3, max_length=1000)
    max_steps: int = Field(default=8, ge=1, le=15)


def default_provider_factory(dataset: Dataset) -> Provider:
    """Real Claude when ANTHROPIC_API_KEY is set; scripted demo only for the sample dataset."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    if dataset.is_sample:
        return demo_provider()
    raise HTTPException(
        status_code=503,
        detail="No ANTHROPIC_API_KEY set. Without a key only the built-in sample dataset "
               "(scripted demo) can be queried.",
    )


def _tables_of(db: Database) -> list[str]:
    if isinstance(db, SQLiteDatabase):
        rows = db.run_select(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).rows
        return sorted(r[0] for r in rows)
    return []


def create_app(
    data_dir: str | Path | None = None,
    provider_factory: Callable[[Dataset], Provider] = default_provider_factory,
) -> FastAPI:
    root = Path(data_dir) if data_dir else Path(tempfile.gettempdir()) / "ask_your_data"
    root.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, Dataset] = {}

    app = FastAPI(title="ask-your-data", version="0.1.0")

    def describe(ds: Dataset) -> dict:
        return {"dataset_id": ds.id, "name": ds.name, "tables": ds.tables,
                "dialect": ds.db.dialect, "is_sample": ds.is_sample}

    def register(name: str, db: Database, is_sample: bool = False) -> Dataset:
        ds = Dataset(uuid.uuid4().hex[:12], name, db, _tables_of(db), is_sample)
        datasets[ds.id] = ds
        return ds

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/datasets", status_code=201)
    def upload_datasets(files: list[UploadFile] = File(...)) -> dict:
        """Upload one or more CSV files; each becomes a table in a private SQLite file."""
        ds_id = uuid.uuid4().hex[:12]
        staging = root / f"{ds_id}_upload"
        staging.mkdir()
        paths: dict[str, Path] = {}
        for up in files:
            if not (up.filename or "").lower().endswith(".csv"):
                raise HTTPException(400, f"{up.filename}: only .csv files are accepted.")
            data = up.file.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{up.filename}: larger than 20 MB.")
            target = staging / f"{sanitize_name(Path(up.filename).stem)}.csv"
            target.write_bytes(data)
            paths[target.stem] = target
        db_path = root / f"{ds_id}.db"
        try:
            ingest_csvs(paths, db_path)
        except Exception as exc:  # malformed CSV: report, do not 500
            raise HTTPException(400, f"Could not read CSV: {exc}") from exc
        finally:
            for p in paths.values():
                p.unlink(missing_ok=True)
            staging.rmdir()
        ds = Dataset(ds_id, ", ".join(paths), SQLiteDatabase(db_path), [])
        ds.tables = _tables_of(ds.db)
        datasets[ds.id] = ds
        return describe(ds)

    @app.post("/datasets/sample", status_code=201)
    def load_sample() -> dict:
        """Create the synthetic textile dataset."""
        ds_id = uuid.uuid4().hex[:12]
        db = SQLiteDatabase(build_sample_db(root / f"{ds_id}.db"))
        ds = Dataset(ds_id, "sample: textile production", db, _tables_of(db), True)
        datasets[ds.id] = ds
        return describe(ds)

    @app.post("/datasets/connect", status_code=201)
    def connect(req: ConnectRequest) -> dict:
        """Point at an existing SQLite/PostgreSQL database. Off by default (see README)."""
        if os.environ.get("ASK_YOUR_DATA_ALLOW_DB_URLS") != "1":
            raise HTTPException(
                403, "Connecting to database URLs is disabled. Set ASK_YOUR_DATA_ALLOW_DB_URLS=1 "
                     "on a trusted, local deployment to enable it.")
        try:
            db = open_database(req.url)
        except (FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return describe(register(req.url.split("@")[-1], db))

    @app.get("/datasets/{dataset_id}")
    def get_dataset(dataset_id: str) -> dict:
        ds = datasets.get(dataset_id)
        if not ds:
            raise HTTPException(404, "Unknown dataset.")
        return describe(ds)

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        ds = datasets.get(req.dataset_id)
        if not ds:
            raise HTTPException(404, "Unknown dataset.")
        provider = provider_factory(ds)
        try:
            result = Agent(provider, ds.db, max_steps=req.max_steps).ask(req.question)
        except ProviderError as exc:
            raise HTTPException(502, str(exc)) from exc
        return {
            "question": result.question,
            "answer": result.answer,
            "stop_reason": result.stop_reason,
            "provider": provider.name,
            "sql": result.sql,
            "steps": [
                {"index": s.index, "tool": s.tool, "input": s.input, "output": s.output,
                 "is_error": s.is_error, "note": s.note}
                for s in result.steps
            ],
            "chart_base64": result.chart_base64,
        }

    return app


def app_factory() -> FastAPI:  # for `uvicorn ask_your_data.api:app_factory --factory`
    return create_app()
