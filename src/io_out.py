from pathlib import Path
import json

import pandas as pd

from src.agent.state import AgentState


MATCH_KEY_FAMILIES_BY_ISSUE_ID = {
    "I1_null_units": "null_rate",
    "I2_duplicate_rows": "duplicates",
    "I3_schema_drift_price": "schema_drift",
    "I4_out_of_range_price": "out_of_range",
    "I5_negative_units": "negative_value",
    "I6_timezone_shift": "mixed_tz",
    "I7_category_typos": "category_variants",
}
MATCH_KEY_FAMILIES_BY_ANOMALY_PREFIX = {
    "null_rate": "null_rate",
    "duplicates": "duplicates",
    "schema_drift": "schema_drift",
    "price_outlier": "out_of_range",
    "negative_units": "negative_value",
    "mixed_tz": "mixed_tz",
    "category_variants": "category_variants",
}


def _match_key(table: object, column: object, signal_family: object) -> str:
    return f"{table}|{column}|{signal_family}"


def export_powerbi(
    state: AgentState,
    ground_truth_path: str | Path = "data/ground_truth.json",
    output_dir: str | Path = "outputs/powerbi",
) -> dict[str, Path]:
    """Write a flat four-table report bundle from a completed agent run."""
    if state.profile is None:
        raise ValueError("A completed profile is required for Power BI export")
    manifest = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    truth_rows = []
    for issue in manifest["issues"]:
        signal_family = MATCH_KEY_FAMILIES_BY_ISSUE_ID[issue["id"]]
        # Duplicates are a table-level finding rather than three column findings.
        columns = ["__table__"] if issue["id"] == "I2_duplicate_rows" else issue["columns"]
        for column in columns:
            truth_rows.append({
                "issue_id": issue["id"], "table_name": issue["table"],
                "column_name": column, "match_key": _match_key(issue["table"], column, signal_family),
                "detection_signal": issue["detection_signal"],
                "affected_row_estimate": issue["affected_row_estimate"],
                "recommended_action": issue["recommended_action"],
                "rationale": issue["rationale"],
            })
    directory = Path(output_dir)
    health_path, proposals_path = write_outputs(state, directory)
    proposals = pd.read_parquet(proposals_path)
    proposals["signal_family"] = proposals["anomaly_id"].map(
        lambda anomaly_id: MATCH_KEY_FAMILIES_BY_ANOMALY_PREFIX.get(str(anomaly_id).split(":", 1)[0])
    )
    proposals["match_key"] = proposals.apply(
        lambda row: _match_key(row["table_name"], row["column_name"], row["signal_family"]),
        axis=1,
    )
    proposals.to_parquet(proposals_path, index=False)
    truth = pd.DataFrame(truth_rows, columns=["issue_id", "table_name", "column_name", "match_key",
                                            "detection_signal", "affected_row_estimate", "recommended_action", "rationale"])
    for column in truth.columns:
        truth[column] = truth[column].astype("int64" if column == "affected_row_estimate" else "string")
    truth_path = directory / "injected_ground_truth.parquet"
    truth.to_parquet(truth_path, index=False)
    summary = pd.DataFrame([{
        "total_issues": len(state.anomalies),
        "high_severity_count": sum(a.severity == "high" for a in state.anomalies),
        "proposals_generated": len(state.proposals),
        "avg_confidence": sum(p.confidence for p in state.surfaced_proposals) / len(state.surfaced_proposals) if state.surfaced_proposals else None,
    }])
    summary["avg_confidence"] = summary["avg_confidence"].astype("float64")
    summary_path = directory / "summary.parquet"
    summary.to_parquet(summary_path, index=False)
    return {"data_health": health_path, "proposals": proposals_path,
            "injected_ground_truth": truth_path, "summary": summary_path}


def write_outputs(state: AgentState, output_dir: str | Path = "outputs") -> tuple[Path, Path]:
    """Export scalar columns only; complex evidence is stored as JSON text."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    profile = state.profile or {"columns": {}}
    health_rows = []
    for column, stats in profile["columns"].items():
        health_rows.append({"table_name": state.table_name, "column_name": column,
                           "row_count": profile["row_count"],
                           "duplicate_row_count": profile["duplicate_row_count"],
                           "schema_hash": profile["schema_hash"],
                           **{key: json.dumps(value, default=str) if isinstance(value, (list, dict)) else value
                              for key, value in stats.items()}})
    health_columns = ["table_name", "column_name", "row_count", "duplicate_row_count", "schema_hash",
                      "dtype", "null_rate", "distinct_count", "min", "max", "mean", "top_values"]
    health = pd.DataFrame(health_rows) if health_rows else pd.DataFrame(columns=health_columns)
    # Different columns have different extrema types; use text for shared fields.
    for field in ("min", "max", "date_min", "date_max"):
        if field in health:
            health[field] = health[field].map(lambda value: None if pd.isna(value) else str(value))
    anomalies = {anomaly.id: anomaly for anomaly in state.anomalies}
    surfaced = {p.anomaly_id for p in state.surfaced_proposals}
    rejected = {entry.proposal.anomaly_id: entry.reason for entry in state.rejected_proposals}
    rows = []
    for proposal in state.proposals:
        anomaly = anomalies[proposal.anomaly_id]
        verdict = state.critic_verdicts.get(proposal.anomaly_id, {})
        status = "rejected" if proposal.anomaly_id in rejected else (
            "surfaced" if verdict.get("passes") else "bypassed" if proposal.anomaly_id in surfaced else "pending")
        rows.append({"table_name": state.table_name, **proposal.model_dump(),
                     "column_name": anomaly.column, "signal": anomaly.signal,
                     "value": json.dumps(anomaly.value, default=str), "severity": anomaly.severity,
                     "evidence_json": json.dumps(anomaly.evidence, default=str),
                     "run_id": state.run_id, "proposal_status": status,
                     "is_rejected": proposal.anomaly_id in rejected,
                     "critic_notes": verdict.get("notes", "Critic bypassed" if status == "bypassed" else "Not reviewed"),
                     "rejection_reason": rejected.get(proposal.anomaly_id),
                     "critic_total": verdict.get("total"), "critic_passes": verdict.get("passes"),
                     **{f"critic_{key}": verdict.get("scores", {}).get(key)
                        for key in ("specificity", "grounding", "safety", "sql_validity")}})
    proposal_columns = ["table_name", "anomaly_id", "action", "rationale", "sql_or_pseudocode",
                        "confidence", "column_name", "signal", "value", "severity", "evidence_json",
                        "run_id", "proposal_status", "is_rejected", "critic_notes", "rejection_reason",
                        "critic_total", "critic_passes", "critic_specificity", "critic_grounding",
                        "critic_safety", "critic_sql_validity"]
    proposals = pd.DataFrame(rows, columns=proposal_columns)
    for field in ("critic_total", "critic_specificity", "critic_grounding", "critic_safety", "critic_sql_validity"):
        proposals[field] = proposals[field].astype("Int64")
    for field in ("critic_passes", "is_rejected"):
        proposals[field] = proposals[field].astype("boolean")
    paths = directory / "data_health.parquet", directory / "proposals.parquet"
    health.to_parquet(paths[0], index=False)
    proposals.to_parquet(paths[1], index=False)
    return paths


def write_agent_output(records: list[dict[str, object]], output_path: str | Path) -> Path:
    """Write agent findings to Parquet."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_parquet(path, index=False)
    return path
