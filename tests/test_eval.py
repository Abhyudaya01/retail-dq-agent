import json

import pandas as pd
import pytest

from eval.eval import evaluate, evaluate_files


def issue(id, columns, action="investigate", affected=10):
    return {"id": id, "table": "sales_dirty", "columns": columns,
            "recommended_action": action, "affected_row_estimate": affected}


def finding(id, column, action="investigate", passes=True, confidence=.8):
    return {"table_name": "sales_dirty", "anomaly_id": id, "column_name": column,
            "signal": "test", "action": action, "confidence": confidence, "critic_passes": passes}


def test_family_matching_and_conservative_actions():
    frame = pd.DataFrame([
        finding("null_rate:units", "units", "impute", False),
        finding("negative_units:units", "units", "investigate"),
        finding("duplicates:__table__", "__table__", "drop"),
        finding("unresolved_tz:date", "date"),
        finding("category_variants:item_id", "item_id"),
        finding("price_outlier:price_usd", "price_usd"),
    ])
    truth = [issue("I1_null_units", ["units"], "impute"),
             issue("I5_negative_units", ["units"]),
             issue("I2_duplicate_rows", ["item_id", "store_id", "date"], "drop"),
             issue("I6_timezone_shift", ["date"]),
             issue("I7_category_typos", ["cat_id"], "impute"),
             issue("I4_out_of_range_price", ["sell_price", "price_usd"])]
    result = evaluate(frame, truth)
    assert result["recall"] == 4/6 and result["action_precision"] == 1.
    assert result["issues"][0]["detected"] and not result["issues"][0]["critic_passed"]
    assert not result["issues"][3]["detected"]
    assert not result["issues"][4]["detected"]


def test_representative_does_not_cherry_pick_expected_action():
    frame = pd.DataFrame([finding("price_outlier:sell_price", "sell_price", "impute", False, .99),
                          finding("price_outlier:price_usd", "price_usd", "investigate", True, .7)])
    result = evaluate(frame, [issue("I4_out_of_range_price", ["sell_price", "price_usd"], "impute")])
    assert result["recall"] == 1 and result["action_precision"] == 0
    assert result["issues"][0]["anomaly_id"] == "price_outlier:price_usd"
    zero = evaluate(frame, [issue("I4_out_of_range_price", ["sell_price"], affected=0)])
    assert zero["recall"] == 0 and zero["action_precision"] is None


def test_report_and_readme_marker_preservation(tmp_path):
    predictions, truth = tmp_path / "proposals.parquet", tmp_path / "ground_truth.json"
    pd.DataFrame([finding("null_rate:units", "units", "impute")]).to_parquet(predictions)
    truth.write_text(json.dumps({"seed": 42, "issues": [issue("I1_null_units", ["units"], "impute"), issue("I7_category_typos", ["cat_id"], "impute")]}))
    readme, report = tmp_path / "README.md", tmp_path / "report.md"
    readme.write_text("unchanged intro\n<!-- EVAL:START -->\n<RECALL>\n<!-- EVAL:END -->\nunchanged ending\n")
    result = evaluate_files(predictions, truth, report, readme)
    assert result["recall"] == .5
    updated = readme.read_text()
    assert updated.startswith("unchanged intro\n") and updated.endswith("unchanged ending\n")
    assert "I7_category_typos" in updated and "50.0%" in updated
    assert "SHA-256" in report.read_text()
    evaluate_files(predictions, truth, report, readme)
    assert readme.read_text() == updated
    readme.write_text("no markers")
    with pytest.raises(ValueError, match="marker"):
        evaluate_files(predictions, truth, report, readme)
