import json
import random

import pandas as pd
from typer.testing import CliRunner

from src.db import get_conn, load_raw
from src.injection import I7_category_typos, app, corrupt_retail_data


def fixture_frame():
    n = 10000
    return pd.DataFrame({
        "item_id": [f"item_{i}" for i in range(n)],
        "store_id": ["TX_1" if i % 3 == 0 else "CA_1" for i in range(n)],
        "date": pd.date_range("2015-01-01", periods=n, freq="h").normalize(),
        "units": [10] * n, "sell_price": [2.] * n, "cat_id": ["HOBBIES"] * n,
    })


def test_all_injections_and_reproducibility():
    clean = fixture_frame()
    # Include dates on both sides of the schema boundary.
    clean.loc[5000:, "date"] += pd.Timedelta(days=600)
    original = clean.copy(deep=True)
    dirty, truth = corrupt_retail_data(clean)
    source = dirty.iloc[:len(clean)]
    assert len(dirty) == len(clean) + 500
    assert dirty.duplicated().sum() == 500
    assert source.units.isna().sum() == 300
    assert dirty.units.isna().mean() > .005
    assert source.units.lt(0).sum() == 30
    after = clean.date.gt(clean.date.median())
    assert source.loc[after, "sell_price"].isna().all()
    assert source.loc[~after, "price_usd"].isna().all()
    prices = source.sell_price.combine_first(source.price_usd)
    assert prices.eq(200).sum() == 50
    tx = source.store_id.eq("TX_1")
    assert source.loc[tx, "date"].str.endswith("T06:00:00Z").all()
    assert source.loc[~tx, "date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    assert source.cat_id.ne("HOBBIES").sum() == 100
    assert set(source.cat_id) == {"HOBBIES", "Hobbies", "HOBBIES ", "HOBIES"}
    assert len(truth["issues"]) == 7
    assert clean.date.median().isoformat() in truth["issues"][2]["detection_signal"]
    assert [issue["affected_row_estimate"] for issue in truth["issues"]] == [300, 500, int(after.sum()), 50, 30, int(tx.sum()), 100]
    again, manifest = corrupt_retail_data(clean)
    pd.testing.assert_frame_equal(dirty, again)
    assert manifest == truth
    pd.testing.assert_frame_equal(clean, original)


def test_optional_category():
    assert I7_category_typos(fixture_frame().drop(columns="cat_id"), random.Random(42)) == 0


def test_loader_and_cli(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    sales = []
    for store in ("CA_1", "CA_2", "TX_1", "WI_1"):
        for i in range(201):
            sales.append({"item_id": f"item_{i:03}", "store_id": store,
                          "cat_id": "HOBBIES", "d_1": 1, "d_2": i + 1, "d_3": i + 1})
    pd.DataFrame(sales).to_csv(raw / "sales_train_evaluation.csv", index=False)
    pd.DataFrame({"d": ["d_1", "d_2", "d_3"], "date": ["2014-12-31", "2015-01-01", "2016-06-02"], "wm_yr_wk": [1, 2, 3]}).to_csv(raw / "calendar.csv", index=False)
    pd.DataFrame([{"item_id": row["item_id"], "store_id": row["store_id"], "wm_yr_wk": wk, "sell_price": 2.} for row in sales for wk in (1, 2, 3)]).to_csv(raw / "sell_prices.csv", index=False)
    db = tmp_path / "retail.duckdb"
    assert load_raw(db, raw) == 1200
    with get_conn(db) as conn:
        clean = conn.table("sales_clean").fetchdf()
        assert clean.item_id.nunique() == 200
        assert "item_000" not in set(clean.item_id)
        assert set(clean.store_id) == {"CA_1", "CA_2", "TX_1"}
        assert clean.sell_price.eq(2).all()
    manifest = tmp_path / "ground_truth.json"
    result = CliRunner().invoke(app, ["--db", str(db), "--ground-truth", str(manifest)])
    assert result.exit_code == 0, result.exception
    assert len(json.loads(manifest.read_text())["issues"]) == 7
    with get_conn(db) as conn:
        assert conn.table("sales_dirty").count("*").fetchone()[0] == 1700
        types = {row[0]: row[1] for row in conn.execute("DESCRIBE sales_dirty").fetchall()}
        assert types["date"] == "VARCHAR"
