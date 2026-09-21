"""Seeded M5 corruption without model calls."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import typer

from src.db import get_conn, load_raw

app = typer.Typer(help="Inject seven reproducible M5 quality issues.")


def _sample(frame, rng, rate, eligible):
    candidates = list(frame.index[eligible])
    return rng.sample(candidates, min(len(candidates), round(len(frame) * rate)))


def I1_null_units(frame, rng):
    """I1_null_units: missing units, detected by null_rate > 0.005."""
    rows = _sample(frame, rng, .03, frame.units.notna())
    frame.loc[rows, "units"] = float("nan")
    return len(rows)


def I2_duplicate_rows(frame, rng):
    """I2_duplicate_rows: 500 verbatim copies; repeated complete rows/natural keys."""
    if len(frame) < 500:
        raise ValueError("At least 500 rows are required for 500 distinct duplicates")
    rows = rng.sample(list(frame.index), 500)
    return pd.concat([frame, frame.loc[rows]], ignore_index=True)


def I3_schema_drift_price(frame, rng):
    """I3_schema_drift_price: sell_price null and price_usd filled after the median date."""
    dates = pd.to_datetime(frame.date)
    mask = dates > dates.median()
    frame["price_usd"] = frame.sell_price.where(mask)
    frame.loc[mask, "sell_price"] = float("nan")
    return int(mask.sum())


def I4_out_of_range_price(frame, rng):
    """I4_out_of_range_price: 100x prices, detected as extreme price outliers."""
    rows = _sample(frame, rng, .005, frame.sell_price.gt(0))
    frame.loc[rows, "sell_price"] *= 100
    return len(rows)


def I5_negative_units(frame, rng):
    """I5_negative_units: sign-flipped positive units, detected by units < 0."""
    rows = _sample(frame, rng, .003, frame.units.gt(0))
    frame.loc[rows, "units"] *= -1
    return len(rows)


def I6_timezone_shift(frame, rng):
    """I6_timezone_shift: TX_1 +6h UTC strings mixed with naive date strings."""
    dates = pd.to_datetime(frame.date)
    mask = frame.store_id.eq("TX_1")
    frame["date"] = dates.dt.strftime("%Y-%m-%d")
    frame.loc[mask, "date"] = (dates[mask] + pd.Timedelta(hours=6)).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return int(mask.sum())


def I7_category_typos(frame, rng):
    """I7_category_typos: Hobbies/HOBBIES /HOBIES outside the category vocabulary."""
    if "cat_id" not in frame:
        return 0
    rows = _sample(frame, rng, .01, frame.cat_id.eq("HOBBIES"))
    for i, row in enumerate(rows):
        frame.loc[row, "cat_id"] = ("Hobbies", "HOBBIES ", "HOBIES")[i % 3]
    return len(rows)


def corrupt_retail_data(frame: pd.DataFrame, seed: int = 42):
    """Return dirty rows and issue manifest; counts exclude subsequent duplicates."""
    dirty = frame.copy().reset_index(drop=True)
    rng = random.Random(seed)
    drift_cutoff = pd.to_datetime(dirty.date).median().isoformat()
    counts = {}
    # Move prices after introducing outliers; duplicate fully corrupted rows last.
    for func in (I1_null_units, I4_out_of_range_price, I5_negative_units,
                 I3_schema_drift_price, I6_timezone_shift, I7_category_typos):
        counts[func.__name__] = func(dirty, rng)
    dirty = I2_duplicate_rows(dirty, rng)
    counts["I2_duplicate_rows"] = 500
    specs = [
        ("I1_null_units", ["units"], "null_rate > 0.005", "impute", "Missing units distort sales totals."),
        ("I2_duplicate_rows", ["item_id", "store_id", "date"], "duplicate natural keys or complete rows", "drop", "Repeated records double-count sales."),
        ("I3_schema_drift_price", ["sell_price", "price_usd"], f"sell_price null and price_usd populated after {drift_cutoff} (source median date)", "schema_fix", "Consolidate migrated price columns (sell_price and price_usd) into a single canonical column; downstream consumers should not have to know the cutoff date."),
        ("I4_out_of_range_price", ["sell_price", "price_usd"], "price approximately 100x item/store baseline", "investigate", "Extreme prices simulate data-entry errors."),
        ("I5_negative_units", ["units"], "units < 0", "investigate", "Negative sales require source validation."),
        ("I6_timezone_shift", ["date"], "TX_1 dates end in T06:00:00Z; others are naive", "normalize", "TX_1 dates carry UTC+6 suffixes while other stores are naive; normalize to a single timezone convention before any date-based join."),
        ("I7_category_typos", ["cat_id"], "cat_id in ['Hobbies', 'HOBBIES ', 'HOBIES']", "normalize", "Map noncanonical labels ('Hobbies', 'HOBBIES ', 'HOBIES') to the canonical uppercase form; this is label correction, not imputation."),
    ]
    return dirty, {"seed": seed, "count_basis": "injected source rows before duplication", "issues": [
        {"id": issue, "table": "sales_dirty", "columns": columns,
         "detection_signal": signal, "affected_row_estimate": counts[issue],
         "recommended_action": action, "rationale": rationale}
        for issue, columns, signal, action, rationale in specs
    ]}


@app.command()
def main(
    db: Path = typer.Option(Path("data/retail.duckdb")),
    raw_dir: Path = typer.Option(Path("data/raw")),
    ground_truth: Path = typer.Option(Path("data/ground_truth.json")),
    seed: int = typer.Option(42),
    load: bool = typer.Option(False, "--load-raw"),
) -> None:
    if load:
        load_raw(db, raw_dir)
    with get_conn(db) as conn:
        frame = conn.execute("SELECT * FROM sales_clean ORDER BY store_id, item_id, date").fetchdf()
        dirty, manifest = corrupt_retail_data(frame, seed)
        conn.register("dirty_frame", dirty)
        conn.execute("CREATE OR REPLACE TABLE sales_dirty AS SELECT * FROM dirty_frame")
    ground_truth.parent.mkdir(parents=True, exist_ok=True)
    ground_truth.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"Wrote {len(dirty)} rows to sales_dirty and {ground_truth}")


if __name__ == "__main__":
    app()
