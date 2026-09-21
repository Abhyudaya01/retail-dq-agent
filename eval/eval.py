"""Issue-level evaluation with explicit, conservative signal-family matching."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Evaluate issue detection and remediation actions.")
console = Console()

# Keys on the left are actual rule-generated anomaly ID prefixes, not fuzzy text.
SIGNAL_FAMILIES = {
    "null_rate": "missingness", "duplicates": "duplication",
    "schema_drift": "schema_drift", "price_outlier": "price_outlier",
    "negative_units": "negative_units", "mixed_tz": "mixed_timezone",
    "category_variants": "category_typos",
}
GROUND_TRUTH_FAMILIES = {
    "I1_null_units": "missingness", "I2_duplicate_rows": "duplication",
    "I3_schema_drift_price": "schema_drift", "I4_out_of_range_price": "price_outlier",
    "I5_negative_units": "negative_units", "I6_timezone_shift": "mixed_timezone",
    "I7_category_typos": "category_typos",
}
MISS_REASONS = {
    "I1_null_units": "Null rate did not exceed the threshold or no matching units finding was exported.",
    "I2_duplicate_rows": "No matching table-level complete-row duplicate finding was exported.",
    "I3_schema_drift_price": "No matching two-price-column schema finding was exported.",
    "I4_out_of_range_price": "Global price variance can mask outliers; no price z-score above six was exported.",
    "I5_negative_units": "No matching units < 0 finding was exported; zero or missing units cannot be sign-flipped.",
    "I6_timezone_shift": "No mixed-suffix date finding was exported; lexical checks cannot prove a six-hour semantic shift.",
    "I7_category_typos": "Typo variants may be outside the top five categories, or no eligible HOBBIES rows existed.",
}


def evaluate(predictions: pd.DataFrame, issues: list[dict]) -> dict:
    required = {"table_name", "anomaly_id", "column_name", "signal", "action", "confidence", "critic_passes"}
    if not required <= set(predictions.columns):
        raise ValueError(f"Missing proposal fields: {sorted(required - set(predictions.columns))}")
    if len({issue["id"] for issue in issues}) != len(issues):
        raise ValueError("Ground-truth IDs must be unique")
    records = predictions.to_dict("records")
    rows = []
    for issue in issues:
        family = GROUND_TRUTH_FAMILIES[issue["id"]]
        columns = {"__table__"} if family == "duplication" else set(issue["columns"])
        candidates = [record for record in records
                      if record["table_name"] == issue["table"]
                      and record["column_name"] in columns
                      and SIGNAL_FAMILIES.get(str(record["anomaly_id"]).split(":", 1)[0]) == family]
        # Pick one representative without consulting the expected action:
        # approved first, then confidence descending, then stable ID tie-break.
        def passed(record):
            value = record["critic_passes"]
            return False if pd.isna(value) else bool(value)
        candidates.sort(key=lambda r: (-int(passed(r)),
                                      -float(r["confidence"]) if pd.notna(r["confidence"]) else 0.,
                                      str(r["anomaly_id"])))
        landed = issue["affected_row_estimate"] > 0
        chosen = candidates[0] if candidates and landed else None
        expected = set(issue["recommended_action"].split("|"))
        rows.append({"issue_id": issue["id"], "signal_family": family,
                     "columns": ", ".join(sorted(columns)), "affected_rows": issue["affected_row_estimate"],
                     "detected": chosen is not None,
                     "action_correct": bool(chosen and chosen["action"] in expected),
                     "critic_passed": bool(chosen and passed(chosen)),
                     "anomaly_id": chosen["anomaly_id"] if chosen else "-",
                     "proposed_action": chosen["action"] if chosen else "-",
                     "recommended_action": issue["recommended_action"],
                     "notes": ("Injection did not land (zero affected rows); retained in denominator." if not landed
                               else "" if chosen else MISS_REASONS[issue["id"]])})
    detected = sum(row["detected"] for row in rows)
    correct = sum(row["action_correct"] for row in rows)
    return {"total": len(rows), "detected": detected, "correct_actions": correct,
            "recall": detected / len(rows) if rows else None,
            "action_precision": correct / detected if detected else None,
            "issues": rows}


def results_markdown(result):
    recall = f"{result['recall']:.1%}" if result["recall"] is not None else "N/A"
    precision = f"{result['action_precision']:.1%}" if result["action_precision"] is not None else "N/A"
    lines = ["| Metric | Result |", "| --- | --- |",
             f"| Recall | {recall} ({result['detected']}/{result['total']}) |",
             f"| Action precision | {precision} ({result['correct_actions']}/{result['detected']}) |", "",
             "| Injected issue | Detected | Action correct | Critic passed | Proposed / expected action |",
             "| --- | --- | --- | --- | --- |"]
    for row in result["issues"]:
        flags = ["yes" if row[key] else "no" for key in ("detected", "action_correct", "critic_passed")]
        lines.append(f"| {row['issue_id']} | {' | '.join(flags)} | {row['proposed_action']} / {row['recommended_action'].replace('|', ', ')} |")
    misses = [row for row in result["issues"] if not row["detected"]]
    lines += ["", "Missed injected issues: " + (", ".join(row["issue_id"] for row in misses) if misses else "none in this run.")]
    for row in misses:
        lines.append(f"- `{row['issue_id']}`: {row['notes']}")
    return "\n".join(lines)


def evaluate_files(predictions=Path("outputs/proposals.parquet"),
                   ground_truth=Path("data/ground_truth.json"),
                   report=Path("eval/report.md"), readme=Path("README.md")):
    predictions, ground_truth, report, readme = map(Path, (predictions, ground_truth, report, readme))
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- EVAL:START -->", "<!-- EVAL:END -->"
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) >= text.index(end):
        raise ValueError("README must have exactly one ordered EVAL marker block")
    frame = pd.read_parquet(predictions)
    if "run_id" in frame and frame.run_id.dropna().nunique() > 1:
        raise ValueError("Evaluate one run at a time")
    manifest = json.loads(ground_truth.read_text(encoding="utf-8"))
    result = evaluate(frame, manifest["issues"])
    markdown = results_markdown(result)
    table = Table(title="Issue-level evaluation")
    for name in ("Issue", "Detected", "Action correct", "Critic passed", "Action"):
        table.add_column(name)
    for row in result["issues"]:
        table.add_row(row["issue_id"], *[str(row[k]) for k in ("detected", "action_correct", "critic_passed")], row["proposed_action"])
    console.print(table)
    console.print(f"Recall: {result['recall']}; action precision: {result['action_precision']}")
    details = ["# Evaluation Report", "", markdown, "", "## Method", "",
               "Detection includes all candidates, even critic-rejected ones. Critic passed is assessed on the same representative proposal as action correctness.",
               "Matches require table + column + rule signal family. Duplicates use __table__; multi-column issues match any listed column.",
               "Representative selection: critic-approved first, highest confidence next, then anomaly ID. Expected actions never influence selection.",
               "Actions match the manifest enum exactly (or one explicit pipe-delimited alternative); normalize/schema_fix are not silently counted as impute/investigate.",
               "Recall denominator is all manifest entries, including zero-affected injections. Action precision denominator is detected issues; no detections yields N/A.",
               "These are issue-level metrics, not row-level localization or finding precision. Additional false positives do not reduce action precision.",
               "A family match is evidence of a rule firing, not proof of injection causality. Inherent clean-data defects can produce the same signal.",
               "Unknown anomaly prefixes (including unresolved_tz) do not count as detections.", "",
               "## Explicit Family Maps", "", "```json", json.dumps({"anomaly_prefix": SIGNAL_FAMILIES, "ground_truth_id": GROUND_TRUTH_FAMILIES}, indent=2), "```", "",
               "## Provenance", "", f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
               f"Manifest seed: {manifest.get('seed', 'not recorded')}",
               f"Candidate rows: {len(frame)}; run IDs: {frame.run_id.dropna().unique().tolist() if 'run_id' in frame else 'not recorded'}"]
    for path in (predictions, ground_truth):
        details.append(f"- `{path}` SHA-256: `{hashlib.sha256(path.read_bytes()).hexdigest()}`")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(details) + "\n", encoding="utf-8")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    readme.write_text(before + start + "\n" + markdown + "\n" + end + after, encoding="utf-8")
    return result


@app.command()
def main(predictions: Path = Path("outputs/proposals.parquet"),
         ground_truth: Path = Path("data/ground_truth.json"),
         report: Path = Path("eval/report.md"), readme: Path = Path("README.md")):
    evaluate_files(predictions, ground_truth, report, readme)


if __name__ == "__main__":
    app()
