"""Synthetic textile-dyeing data (fixed seed). No real client data is involved.

Tables: machines, production_orders, dye_batches. One machine (JET-07) is built to have a
noticeably higher re-dye rate so the demo question has a real, checkable answer.
"""

from __future__ import annotations

import csv
import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

SEED = 42
N_ORDERS = 600

MACHINES = [
    # id, type, capacity_kg, plant, redye probability
    ("JET-01", "jet", 300, "A", 0.05),
    ("JET-02", "jet", 300, "A", 0.06),
    ("JET-03", "jet", 450, "A", 0.05),
    ("JET-04", "jet", 450, "B", 0.06),
    ("JET-05", "jet", 600, "B", 0.06),
    ("JET-06", "jet", 600, "B", 0.05),
    ("JET-07", "jet", 450, "B", 0.35),  # the "problem" machine
    ("OVF-01", "overflow", 250, "A", 0.07),
    ("OVF-02", "overflow", 250, "A", 0.06),
    ("PKG-01", "package", 800, "B", 0.04),
    ("PKG-02", "package", 800, "B", 0.05),
    ("CNT-01", "continuous", 2000, "A", 0.06),
]
CUSTOMERS = [f"Customer {i:02d}" for i in range(1, 16)]
FABRICS = ["cotton jersey", "cotton poplin", "polyester microfiber", "viscose", "poly-cotton"]
COLORS = ["navy", "black", "red", "forest green", "sand", "white", "royal blue", "burgundy",
          "grey melange", "mustard"]

TABLES = {
    "machines": ["machine_id", "machine_type", "capacity_kg", "plant"],
    "production_orders": ["order_id", "customer", "fabric", "quantity_kg", "order_date",
                          "due_date", "completed_date", "status", "lead_time_days"],
    "dye_batches": ["batch_id", "order_id", "machine_id", "color", "start_time",
                    "duration_hours", "weight_kg", "redye"],
}


def generate() -> dict[str, list[tuple]]:
    """Return table name -> rows. Deterministic for a given SEED."""
    rng = random.Random(SEED)
    redye_p = {m[0]: m[4] for m in MACHINES}
    machines = [(m[0], m[1], m[2], m[3]) for m in MACHINES]
    cap = {m[0]: m[2] for m in MACHINES}

    orders: list[tuple] = []
    batches: list[tuple] = []
    batch_id = 1
    start = date(2025, 1, 6)
    for i in range(1, N_ORDERS + 1):
        order_date = start + timedelta(days=rng.randint(0, 330))
        fabric = rng.choice(FABRICS)
        n_batches = rng.choices([1, 2, 3, 4], weights=[45, 30, 17, 8])[0]
        redyes = 0
        total_kg = 0
        for _ in range(n_batches):
            machine = rng.choice([m[0] for m in MACHINES])
            weight = round(cap[machine] * rng.uniform(0.55, 0.95), 1)
            redye = int(rng.random() < redye_p[machine])
            redyes += redye
            total_kg += weight
            started = datetime.combine(order_date, datetime.min.time()) + timedelta(
                days=rng.randint(1, 4), hours=rng.randint(6, 20))
            duration = round(rng.uniform(5.0, 9.0) + (3.5 if redye else 0), 1)
            batches.append((batch_id, i, machine, rng.choice(COLORS),
                            started.strftime("%Y-%m-%d %H:%M"), duration, weight, redye))
            batch_id += 1
        lead = rng.randint(4, 9) + n_batches // 2 + 2 * redyes
        due = order_date + timedelta(days=rng.randint(7, 12))
        in_progress = order_date > date(2025, 11, 10)
        completed = None if in_progress else order_date + timedelta(days=lead)
        orders.append((
            i, rng.choice(CUSTOMERS), fabric, round(total_kg), order_date.isoformat(),
            due.isoformat(), completed.isoformat() if completed else None,
            "in_progress" if in_progress else "completed", None if in_progress else lead,
        ))
    return {"machines": machines, "production_orders": orders, "dye_batches": batches}


def write_csvs(out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for table, rows in generate().items():
        path = out / f"{table}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(TABLES[table])
            writer.writerows(rows)
        paths.append(path)
    return paths


def build_sample_db(db_path: str | Path) -> Path:
    """Create a SQLite file with the sample tables (overwrites)."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    types = {
        "machines": ["TEXT", "TEXT", "INTEGER", "TEXT"],
        "production_orders": ["INTEGER", "TEXT", "TEXT", "INTEGER", "TEXT", "TEXT", "TEXT",
                              "TEXT", "INTEGER"],
        "dye_batches": ["INTEGER", "INTEGER", "TEXT", "TEXT", "TEXT", "REAL", "REAL", "INTEGER"],
    }
    conn = sqlite3.connect(str(path))
    try:
        for table, rows in generate().items():
            cols = ", ".join(f"{c} {t}" for c, t in zip(TABLES[table], types[table], strict=True))
            conn.execute(f"CREATE TABLE {table} ({cols})")
            marks = ",".join("?" * len(TABLES[table]))
            conn.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)
        conn.commit()
    finally:
        conn.close()
    return path


if __name__ == "__main__":  # python -m ask_your_data.sample_data [out_dir]
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "sample_data"
    for p in write_csvs(target):
        print("wrote", p)
