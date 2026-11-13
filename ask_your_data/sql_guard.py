"""SQL guard: only a single read-only SELECT may pass.

This is layer 1 of 2. Layer 2 is the read-only connection itself (see db.py), so a bug here
does not turn into a write.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError


class SQLGuardError(ValueError):
    """Raised when a statement is not an allowed read-only query."""


# Node types that mean "this statement changes state or escapes the query".
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name)
    for name in (
        "Insert", "Update", "Delete", "Merge", "Drop", "Create", "Alter", "AlterTable",
        "TruncateTable", "Command", "Pragma", "Attach", "Detach", "Into", "Lock", "Copy",
        "Set", "Use", "Transaction", "Commit", "Rollback", "Grant",
    )
    if hasattr(exp, name)
)

# Functions with side effects or file/network access (SQLite + PostgreSQL).
_FORBIDDEN_FUNCTIONS = frozenset(
    {
        "load_extension", "readfile", "writefile", "pg_sleep", "pg_read_file",
        "pg_read_binary_file", "pg_ls_dir", "lo_import", "lo_export", "dblink",
        "dblink_exec", "set_config", "pg_terminate_backend", "pg_cancel_backend",
    }
)


def validate_select(sql: str, dialect: str = "sqlite") -> str:
    """Return the cleaned SQL if it is exactly one read-only SELECT, else raise."""
    if not sql or not sql.strip():
        raise SQLGuardError("Empty query.")
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except SqlglotError as exc:
        raise SQLGuardError(f"Could not parse SQL: {exc}") from exc

    if len(statements) != 1:
        raise SQLGuardError("Exactly one statement is allowed.")
    stmt = statements[0]

    if not isinstance(stmt, exp.Query):
        raise SQLGuardError(f"Only SELECT queries are allowed (got {type(stmt).__name__}).")

    for node in stmt.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SQLGuardError(f"Forbidden construct in query: {type(node).__name__}.")
        if isinstance(node, exp.Anonymous) and str(node.this).lower() in _FORBIDDEN_FUNCTIONS:
            raise SQLGuardError(f"Forbidden function: {node.this}.")
        if isinstance(node, exp.Func) and node.sql_name().lower() in _FORBIDDEN_FUNCTIONS:
            raise SQLGuardError(f"Forbidden function: {node.sql_name()}.")

    return sql.strip().rstrip(";").strip()
