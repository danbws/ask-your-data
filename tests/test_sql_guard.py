import pytest

from ask_your_data.db import QueryError, SQLiteDatabase
from ask_your_data.sql_guard import SQLGuardError, validate_select


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM machines",
        "select machine_id, count(*) from dye_batches group by 1;",
        "WITH r AS (SELECT machine_id, SUM(redye) s FROM dye_batches GROUP BY 1) SELECT * FROM r",
        "SELECT 1 UNION SELECT 2",
        "SELECT * FROM (SELECT 1 AS a) t",
    ],
)
def test_allows_read_only_selects(sql):
    assert validate_select(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO machines VALUES ('X', 'jet', 1, 'A')",
        "UPDATE machines SET plant = 'Z'",
        "DELETE FROM machines",
        "DROP TABLE machines",
        "CREATE TABLE t (a int)",
        "ALTER TABLE machines ADD COLUMN x int",
        "PRAGMA writable_schema = 1",
        "ATTACH DATABASE 'x.db' AS x",
        "SELECT 1; DROP TABLE machines",
        "WITH x AS (SELECT 1) DELETE FROM machines",
        "SELECT load_extension('evil')",
        "",
        "   ",
        "this is not sql",
    ],
)
def test_rejects_writes_ddl_and_junk(sql):
    with pytest.raises(SQLGuardError):
        validate_select(sql)


def test_postgres_specific_constructs_rejected():
    with pytest.raises(SQLGuardError):
        validate_select("SELECT pg_sleep(100)", dialect="postgres")
    with pytest.raises(SQLGuardError):
        validate_select("SELECT * INTO newt FROM machines", dialect="postgres")


def test_guard_blocks_before_touching_the_database(db):
    with pytest.raises(SQLGuardError):
        db.run_select("DELETE FROM machines")
    assert db.run_select("SELECT COUNT(*) FROM machines").rows == [(12,)]


def test_connection_is_read_only_even_if_the_guard_were_bypassed(sample_db_path):
    """Layer 2: call the raw connection with a write; SQLite itself must refuse."""
    import sqlite3

    conn = SQLiteDatabase(sample_db_path)._connect()
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM machines")
    conn.close()


def test_row_limit_and_truncation_flag(db):
    result = db.run_select("SELECT * FROM dye_batches", row_limit=10)
    assert len(result.rows) == 10
    assert result.truncated


def test_timeout_aborts_runaway_query(sample_db_path):
    db = SQLiteDatabase(sample_db_path, timeout_s=0.3)
    slow = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 50000000) "
        "SELECT COUNT(*) FROM c"
    )
    with pytest.raises(QueryError, match="timeout"):
        db.run_select(slow)
