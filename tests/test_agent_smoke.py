import json
from unittest.mock import Mock

import pandas as pd
import pyarrow.parquet as pq

from src.agent.graph import run
from src.agent.nodes import anomaly_flagger_node, profiler_node
from src.agent.state import AgentState, Proposal
from src.db import get_conn
from src.io_out import export_powerbi, write_outputs


def test_graph_seeded_issues_with_mock_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-mocked-key")
    n = 250
    frame = pd.DataFrame({"item_id": [f"item_{i}" for i in range(n)],
                          "store_id": ["CA_1"] * n, "units": [5.] * n,
                          "sell_price": [2.] * n, "price_usd": [None] * n,
                          "date": ["2016-06-01"] * n, "cat_id": ["HOBBIES"] * n})
    frame.loc[:9, "units"] = None
    frame.loc[10, "units"] = -5
    frame.loc[11, "sell_price"] = 200
    frame.loc[12:20, "sell_price"] = None
    frame.loc[12:20, "price_usd"] = 2.
    frame.loc[12:20, "date"] = "2016-06-02T06:00:00Z"
    frame.loc[21, "cat_id"] = "Hobbies"
    frame.loc[22, "cat_id"] = "HOBBIES "
    frame.loc[23, "cat_id"] = "HOBIES"
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    db = tmp_path / "fixture.duckdb"
    with get_conn(db) as conn:
        conn.register("fixture", frame)
        conn.execute("CREATE TABLE sales_dirty AS SELECT * FROM fixture")
    llm = Mock()

    def propose(messages):
        anomaly = json.loads(messages[-1][1])
        return Proposal(anomaly_id=anomaly["id"], action="investigate",
                        rationale="Verify the retail source before modifying daily sales or USD shelf prices.",
                        sql_or_pseudocode="Review flagged records in sales_dirty.", confidence=.8)

    llm.invoke.side_effect = propose
    monkeypatch.setattr("src.agent.nodes.get_proposal_llm", lambda: llm)
    state = run(db_path=db, use_critic=False)
    kinds = {a.id.split(":")[0] for a in state.anomalies}
    assert {"null_rate", "duplicates", "negative_units", "price_outlier", "mixed_tz", "schema_drift"} <= kinds
    assert "category_variants" in kinds
    assert {a.column for a in state.anomalies if a.id.startswith("category_variants:")} == {"cat_id"}
    assert len(state.proposals) == len(state.anomalies) == llm.invoke.call_count
    assert {p.anomaly_id for p in state.proposals} == {a.id for a in state.anomalies}
    assert state.profile["columns"]["date"]["tz_check_confidence"] == "low"
    assert state.profile["columns"]["units"]["mean"] is not None
    health_path, proposals_path = write_outputs(state, tmp_path / "outputs")
    assert len(pd.read_parquet(health_path)) == len(frame.columns)
    assert len(pd.read_parquet(proposals_path)) == len(state.proposals)
    for path in (health_path, proposals_path):
        assert all(not field.type.__class__.__name__.startswith(("List", "Struct", "Map")) for field in pq.read_schema(path))
    assert pd.read_parquet(proposals_path).evidence_json.map(json.loads).map(type).eq(dict).all()
    manifest = tmp_path / "ground_truth.json"
    manifest.write_text(json.dumps({"issues": [
        {"id": "I1_null_units", "table": "sales_dirty", "columns": ["units"],
         "detection_signal": "null_rate > 0.005", "affected_row_estimate": 10,
         "recommended_action": "investigate", "rationale": "Missing daily sales."},
        {"id": "I2_duplicate_rows", "table": "sales_dirty", "columns": ["item_id", "store_id", "date"],
         "detection_signal": "duplicates", "affected_row_estimate": 1,
         "recommended_action": "drop", "rationale": "Repeated sales."},
        {"id": "I3_schema_drift_price", "table": "sales_dirty", "columns": ["sell_price", "price_usd"],
         "detection_signal": "drift", "affected_row_estimate": 9,
         "recommended_action": "investigate", "rationale": "Price migration."},
    ]}))
    bundle = export_powerbi(state, manifest, tmp_path / "powerbi")
    assert len(bundle) == 4
    truth = pd.read_parquet(bundle["injected_ground_truth"])
    assert len(truth) == 4 and truth.issue_id.nunique() == 3
    assert truth.match_key.is_unique
    assert truth.loc[truth.issue_id.eq("I2_duplicate_rows"), "match_key"].item() == "sales_dirty|__table__|duplicates"
    proposals = pd.read_parquet(bundle["proposals"])
    assert proposals.loc[proposals.anomaly_id.eq("duplicates:__table__"), "match_key"].item() == "sales_dirty|__table__|duplicates"
    assert proposals.loc[proposals.anomaly_id.eq("schema_drift:sell_price"), "match_key"].item() == "sales_dirty|sell_price|schema_drift"
    summary = pd.read_parquet(bundle["summary"]).iloc[0]
    assert summary.total_issues == len(state.anomalies)
    assert summary.high_severity_count == sum(a.severity == "high" for a in state.anomalies)
    assert summary.proposals_generated == len(state.proposals)
    assert abs(summary.avg_confidence - .8) < 1e-10


def test_unresolved_dates_are_not_skipped(tmp_path):
    db = tmp_path / "dates.duckdb"
    with get_conn(db) as conn:
        conn.execute("CREATE TABLE sales_dirty AS SELECT 'unknown date' AS date")
    state = AgentState()
    profiler_node(state, db)
    anomaly_flagger_node(state)
    assert any(a.id == "unresolved_tz:date" and a.severity == "low" for a in state.anomalies)


def test_empty_table_no_model_call(tmp_path, monkeypatch):
    db = tmp_path / "empty.duckdb"
    with get_conn(db) as conn:
        conn.execute("CREATE TABLE sales_dirty (units INTEGER, date VARCHAR)")
    factory = Mock(side_effect=AssertionError("Empty table must not call LLM"))
    monkeypatch.setattr("src.agent.nodes.get_proposal_llm", factory)
    state = run(db_path=db)
    assert state.proposals == []
    write_outputs(state, tmp_path / "empty_outputs")
    factory.assert_not_called()
