"""Database access: read-only SQLite (default) and PostgreSQL (optional extra)."""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .sql_guard import validate_select

DEFAULT_ROW_LIMIT = 200
DEFAULT_TIMEOUT_S = 5.0


class QueryError(RuntimeError):
    """A query failed (bad SQL, timeout, ...). The message is safe to show to the model."""


def _fmt(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value).replace("|", "/").replace("\n", " ")


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool = False
    elapsed_ms: int = 0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)

    def to_text(self, max_rows: int = 100) -> str:
        """Compact markdown table: what the model (and the trace) sees."""
        shown = self.rows[:max_rows]
        lines = [
            "| " + " | ".join(self.columns) + " |",
            "|" + "|".join("---" for _ in self.columns) + "|",
        ]
        for row in shown:
            lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
        note = f"{len(self.rows)} row(s)"
        if self.truncated:
            note += " (truncated at the row limit; add aggregation/filters for full results)"
        elif len(self.rows) > max_rows:
            note += f", showing first {max_rows}"
        return "\n".join(lines) + f"\n\n{note}"


class SQLiteDatabase:
    """Read-only SQLite access.

    Defense in depth: the SQL is parsed by the guard, the file is opened with ``mode=ro``,
    an authorizer denies anything but reads, and a progress handler enforces the timeout.
    """

    dialect = "sqlite"

    def __init__(
        self,
        path: str | Path,
        row_limit: int = DEFAULT_ROW_LIMIT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"SQLite file not found: {self.path}")
        self.row_limit = row_limit
        self.timeout_s = timeout_s

    def _connect(self, restricted: bool = True) -> sqlite3.Connection:
        conn = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only = ON")
        if not restricted:  # internal use only (schema introspection), never model-supplied SQL
            return conn
        allowed = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
            sqlite3.SQLITE_RECURSIVE,
        }
        conn.set_authorizer(
            lambda action, *_: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
        )
        return conn

    def run_select(self, sql: str, row_limit: int | None = None) -> QueryResult:
        clean = validate_select(sql, self.dialect)
        limit = row_limit or self.row_limit
        conn = self._connect()
        deadline = time.monotonic() + self.timeout_s
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        started = time.monotonic()
        try:
            cur = conn.execute(clean)
            columns = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(limit + 1)
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise QueryError(f"Query exceeded the {self.timeout_s:.0f}s timeout.") from exc
            raise QueryError(str(exc)) from exc
        except sqlite3.DatabaseError as exc:
            raise QueryError(str(exc)) from exc
        finally:
            conn.close()
        truncated = len(rows) > limit
        return QueryResult(
            columns,
            [tuple(r) for r in rows[:limit]],
            truncated,
            int((time.monotonic() - started) * 1000),
        )

    def schema_description(self, sample_rows: int = 3) -> str:
        conn = self._connect(restricted=False)
        try:
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            parts = []
            for table in tables:
                cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
                count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                col_txt = ", ".join(f"{c[1]} {c[2] or 'TEXT'}" for c in cols)
                parts.append(f"TABLE {table} ({count} rows): {col_txt}")
                for row in conn.execute(f'SELECT * FROM "{table}" LIMIT {sample_rows}'):
                    parts.append("  e.g. " + ", ".join(_fmt(v) for v in row))
            return "\n".join(parts)
        finally:
            conn.close()


class PostgresDatabase:
    """Read-only PostgreSQL access (optional ``postgres`` extra, psycopg 3).

    Read-only is enforced by the guard AND by a read-only session with a statement timeout.
    NOTE: this path is not covered by the automated tests (CI has no PostgreSQL service).
    """

    dialect = "postgres"

    def __init__(
        self,
        url: str,
        row_limit: int = DEFAULT_ROW_LIMIT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        try:
            import psycopg  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PostgreSQL support needs: pip install 'ask-your-data[postgres]'"
            ) from exc
        self.url = url
        self.row_limit = row_limit
        self.timeout_s = timeout_s

    def _connect(self):  # pragma: no cover - needs a server
        import psycopg

        conn = psycopg.connect(
            self.url,
            options=(
                "-c default_transaction_read_only=on "
                f"-c statement_timeout={int(self.timeout_s * 1000)}"
            ),
        )
        conn.read_only = True
        return conn

    def run_select(self, sql: str, row_limit: int | None = None) -> QueryResult:  # pragma: no cover
        clean = validate_select(sql, self.dialect)
        limit = row_limit or self.row_limit
        started = time.monotonic()
        with self._connect() as conn, conn.cursor() as cur:
            try:
                cur.execute(clean)
                columns = [d.name for d in cur.description or []]
                rows = cur.fetchmany(limit + 1)
            except Exception as exc:
                raise QueryError(str(exc)) from exc
        return QueryResult(
            columns,
            [tuple(r) for r in rows[:limit]],
            len(rows) > limit,
            int((time.monotonic() - started) * 1000),
        )

    def schema_description(self, sample_rows: int = 3) -> str:  # pragma: no cover
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT table_name, column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'public' ORDER BY table_name, ordinal_position"
            )
            tables: dict[str, list[str]] = {}
            for table, col, typ in cur.fetchall():
                tables.setdefault(table, []).append(f"{col} {typ}")
        return "\n".join(f"TABLE {t}: {', '.join(c)}" for t, c in tables.items())


Database = SQLiteDatabase | PostgresDatabase


def open_database(url: str, **kwargs) -> Database:
    """Open ``sqlite:///path/to.db``, a plain .db path, or ``postgresql://...``."""
    if url.startswith(("postgresql://", "postgres://")):
        return PostgresDatabase(url, **kwargs)
    path = url.removeprefix("sqlite:///")
    return SQLiteDatabase(path, **kwargs)


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower() or "col"
    return "_" + cleaned if cleaned[0].isdigit() else cleaned


@dataclass
class IngestReport:
    tables: dict[str, int] = field(default_factory=dict)  # table -> row count


def ingest_csvs(files: dict[str, str | Path], db_path: str | Path) -> IngestReport:
    """Load CSV files into a fresh SQLite file. ``files`` maps a name -> CSV path."""
    report = IngestReport()
    conn = sqlite3.connect(str(db_path))
    try:
        for raw_name, csv_path in files.items():
            frame = pd.read_csv(csv_path)
            frame.columns = [sanitize_name(str(c)) for c in frame.columns]
            if len(set(frame.columns)) != len(frame.columns):
                raise ValueError(f"{raw_name}: duplicate column names after cleaning")
            table = sanitize_name(Path(raw_name).stem)
            frame.to_sql(table, conn, index=False, if_exists="replace")
            report.tables[table] = len(frame)
        conn.commit()
    finally:
        conn.close()
    return report
