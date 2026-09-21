import json
from unittest.mock import Mock

import pandas as pd
import pytest

from src.agent.graph import run
from src.agent.state import Proposal
from src.critic import CriticVerdict, critique
from src.db import get_conn
from src.io_out import export_powerbi, write_outputs


def proposal(anomaly_id="null_rate:units"):
    return Proposal(anomaly_id=anomaly_id, action="investigate",
                    rationale="Units are daily sales; reconcile missing records.",
                    sql_or_pseudocode="SELECT * FROM sales_dirty WHERE units IS NULL", confidence=.9)


@pytest.mark.parametrize("scores,total,passes", [([4,4,3,3],14,True), ([4,3,3,3],13,False)])
def test_critique_fixed_rubric(monkeypatch, scores, total, passes):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-mocked-key")
    llm = Mock()
    llm.invoke.return_value = {"scores": dict(zip(("specificity", "grounding", "safety", "sql_validity"), scores)),
                               "total": 20, "passes": True, "notes": "Fixed rubric notes"}
    monkeypatch.setattr("src.critic.get_critic_llm", lambda: llm)
    verdict = critique(proposal(), {"column": "units"})
    assert verdict.total == total and verdict.passes is passes
    llm.invoke.assert_called_once()


def test_routing_persistence_export_and_bypass(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-mocked-key")
    db = tmp_path / "retail.duckdb"
    with get_conn(db) as conn:
        conn.execute("CREATE TABLE sales_dirty AS SELECT i AS row_id, CASE WHEN i < 10 THEN NULL WHEN i = 10 THEN -1 ELSE 5 END AS units FROM range(100) t(i)")
    proposer = Mock()
    proposer.invoke.side_effect = lambda messages: proposal(json.loads(messages[-1][1])["id"])
    monkeypatch.setattr("src.agent.nodes.get_proposal_llm", lambda: proposer)
    critic = Mock()
    def grade(messages):
        record = json.loads(messages[-1][1])["proposal"]
        specificity = 4 if record["anomaly_id"].startswith("null_rate") else 3
        return {"scores": {"specificity": specificity, "grounding": 4, "safety": 3, "sql_validity": 3},
                "total": 20, "passes": True, "notes": "Needs exact row selection"}
    critic.invoke.side_effect = grade
    monkeypatch.setattr("src.critic.get_critic_llm", lambda: critic)
    state = run(db_path=db)
    assert len(state.surfaced_proposals) == len(state.rejected_proposals) == 1
    assert state.rejected_proposals[0].reason == "Needs exact row selection"
    with get_conn(db) as conn:
        rows = conn.execute("SELECT run_id, critic_total, critic_passes, trace_json FROM fixes ORDER BY critic_total").fetchall()
        assert [(row[1],row[2]) for row in rows] == [(13,False),(14,True)]
        assert all(row[0] == state.run_id for row in rows)
        for row in rows:
            trace = json.loads(row[3])
            assert [node["node"] for node in trace["nodes"]] == ["profiler", "flagger", "proposer", "critic"]
            assert all("inputs" in node and "outputs" in node for node in trace["nodes"])
            assert [call["node"] for call in trace["llm_calls"]] == ["proposer", "critic"]
    _, path = write_outputs(state, tmp_path / "outputs")
    exported = pd.read_parquet(path)
    assert set(exported.proposal_status) == {"surfaced", "rejected"}
    assert exported.loc[exported.is_rejected, "critic_notes"].item() == "Needs exact row selection"
    manifest = tmp_path / "ground_truth.json"
    manifest.write_text(json.dumps({"issues": []}))
    bundle = export_powerbi(state, manifest, tmp_path / "powerbi")
    powerbi_proposals = pd.read_parquet(bundle["proposals"])
    assert set(powerbi_proposals.proposal_status) == {"surfaced", "rejected"}
    assert powerbi_proposals.loc[powerbi_proposals.is_rejected, "critic_total"].item() == 13
    assert powerbi_proposals.loc[powerbi_proposals.is_rejected, "critic_notes"].item() == "Needs exact row selection"
    assert pd.read_parquet(bundle["summary"]).avg_confidence.item() == .9
    critic.reset_mock()
    bypassed = run(db_path=db, use_critic=False)
    assert len(bypassed.surfaced_proposals) == 2 and bypassed.rejected_proposals == []
    critic.invoke.assert_not_called()
    with get_conn(db) as conn:
        assert conn.execute("SELECT count(*) FROM fixes").fetchone()[0] == 4
        assert conn.execute("SELECT count(*) FROM fixes WHERE critic_passes IS NULL").fetchone()[0] == 2


def test_invalid_scores_rejected():
    with pytest.raises(ValueError):
        CriticVerdict(scores={"specificity": 6, "grounding": 5, "safety": 5, "sql_validity": 5}, total=20, passes=True, notes="")
