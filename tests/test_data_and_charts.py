import filecmp

import pytest

from ask_your_data.charts import ChartError, render_chart
from ask_your_data.db import SQLiteDatabase, ingest_csvs
from ask_your_data.sample_data import write_csvs


def test_sample_data_is_deterministic(tmp_path):
    a = write_csvs(tmp_path / "a")
    b = write_csvs(tmp_path / "b")
    assert all(filecmp.cmp(x, y, shallow=False) for x, y in zip(a, b, strict=True))


def test_committed_sample_csvs_match_the_generator(tmp_path):
    from pathlib import Path

    fresh = write_csvs(tmp_path)
    committed = Path(__file__).parent.parent / "sample_data"
    for f in fresh:
        assert (committed / f.name).read_bytes() == f.read_bytes(), (
            f"{f.name} is stale: run python -m ask_your_data.sample_data sample_data")


def test_sample_data_has_the_planted_signal(db):
    top = db.run_select(
        "SELECT machine_id FROM dye_batches GROUP BY machine_id "
        "ORDER BY 1.0 * SUM(redye) / COUNT(*) DESC LIMIT 1"
    ).rows[0][0]
    assert top == "JET-07"


def test_csv_ingest_cleans_names(tmp_path):
    csv = tmp_path / "Dye Lots 2025.csv"
    csv.write_text("Lot No,Weight (kg),2nd Pass\n1,10,a\n2,12,b\n", encoding="utf-8")
    ingest_csvs({csv.name: csv}, tmp_path / "d.db")
    db = SQLiteDatabase(tmp_path / "d.db")
    res = db.run_select("SELECT * FROM dye_lots_2025")
    assert res.columns == ["lot_no", "weight_kg", "_2nd_pass"]
    assert len(res.rows) == 2


def test_chart_renders_png(db):
    png = render_chart(db, {
        "kind": "bar", "title": "t",
        "sql": "SELECT plant, COUNT(*) AS n FROM machines GROUP BY plant",
        "x": "plant", "y": ["n"],
    })
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 2000


@pytest.mark.parametrize("spec,msg", [
    ({"kind": "pie", "sql": "SELECT 1"}, "kind"),
    ({"kind": "bar", "sql": "SELECT plant FROM machines", "x": "plant", "y": ["nope"]}, "nope"),
    ({"kind": "bar", "sql": "SELECT plant FROM machines WHERE 0", "x": "plant", "y": ["plant"]},
     "no rows"),
    ({"kind": "bar", "sql": "SELECT plant FROM machines"}, "x and y"),
])
def test_chart_errors_are_clear(db, spec, msg):
    with pytest.raises(ChartError, match=msg):
        render_chart(db, spec)
