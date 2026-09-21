#!/usr/bin/env -S uv run python
"""Clean, execute, and verify the real pipeline without changing agent logic."""
from __future__ import annotations

import math
from pathlib import Path
import re
import subprocess
import sys

from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
DB = ROOT / "data/retail.duckdb"

# Explicit targeted queries: one per injected issue. Counts must be positive.
# Trading-day joins use the date prefix because I6 changes TX_1 date formatting.
CLEAN_JOIN = """JOIN sales_clean c ON d.item_id = c.item_id
    AND d.store_id = c.store_id AND substr(d.date, 1, 10) = CAST(c.date AS VARCHAR)"""
INJECTION_QUERIES = {
    "I1_null_units": f"""SELECT count(*) FROM sales_dirty d {CLEAN_JOIN}
        WHERE d.units IS NULL AND c.units IS NOT NULL""",
    "I2_duplicate_rows": """SELECT count(*) - (SELECT count(*) FROM sales_clean)
        FROM sales_dirty HAVING count(*) - (SELECT count(*) FROM sales_clean) = 500
        AND count(*) - (SELECT count(*) FROM (SELECT DISTINCT * FROM sales_dirty)) >= 500""",
    "I3_schema_drift_price": f"""SELECT count(*) FROM sales_dirty d {CLEAN_JOIN}
        WHERE c.date > (SELECT median(CAST(date AS TIMESTAMP)) FROM sales_clean)
        AND c.sell_price IS NOT NULL
        AND d.sell_price IS NULL AND d.price_usd IS NOT NULL
        AND (abs(d.price_usd - c.sell_price) < 0.000001
             OR abs(d.price_usd - c.sell_price * 100) < 0.000001)""",
    "I4_out_of_range_price": f"""SELECT count(*) FROM sales_dirty d {CLEAN_JOIN}
        WHERE c.sell_price > 0
        AND abs(coalesce(d.sell_price, d.price_usd) - c.sell_price * 100) < 0.000001""",
    "I5_negative_units": f"""SELECT count(*) FROM sales_dirty d {CLEAN_JOIN}
        WHERE c.units > 0 AND d.units = -c.units""",
    "I6_timezone_shift": f"""SELECT count(*) FROM sales_dirty d {CLEAN_JOIN}
        WHERE d.store_id = 'TX_1' AND ends_with(d.date, 'T06:00:00Z')
        AND TRY_CAST(d.date AS TIMESTAMP) = CAST(c.date AS TIMESTAMP) + INTERVAL '6 hours'
        HAVING (SELECT count(*) FROM sales_dirty WHERE store_id <> 'TX_1'
                AND regexp_full_match(date, '[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}')) > 0""",
    "I7_category_typos": f"""SELECT count(*) FROM sales_dirty d {CLEAN_JOIN}
        WHERE c.cat_id = 'HOBBIES' AND d.cat_id IN ('Hobbies', 'HOBBIES ', 'HOBIES')
        HAVING count(DISTINCT d.cat_id) = 3""",
}


def main() -> int:
    console = Console(width=110, force_terminal=False)
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    results = []

    def check(name, operation):
        try:
            ok, detail = operation()
            results.append((name, bool(ok), str(detail)))
        except Exception as error:
            detail = f"{type(error).__name__}: {error}".replace(str(ROOT) + "/", "")
            results.append((name, False, detail.splitlines()[0]))

    def clean():
        paths = [DB, ROOT / "eval/report.md",
                 *(ROOT / "outputs").glob("*.parquet"),
                 *(ROOT / "outputs/powerbi").glob("*.parquet")]
        removed = []
        for path in paths:
            if path.exists():
                path.unlink()
                removed.append(str(path.relative_to(ROOT)))
        return True, f"Removed {len(removed)} requested artifacts; raw CSVs and manifest preserved"

    check("Clean state", clean)
    console.print("Running setup -> inject -> run-agent -> eval; stdout/stderr saved per step.")
    if not results[0][1]:
        console.print("Clean state failed; pipeline commands will not run.")
    else:
        for step in ("setup", "inject", "run-agent", "eval"):
            log = logs / f"verify_{step}.log"

            def command():
                with log.open("w", encoding="utf-8") as stream:
                    process = subprocess.run(["make", step], cwd=ROOT,
                                             stdout=stream, stderr=subprocess.STDOUT, check=False)
                return process.returncode == 0, f"exit={process.returncode}; logs/{log.name}"

            check(f"make {step}", command)
            if step == "inject":
                def sql(query):
                    import duckdb
                    if not DB.is_file():
                        raise FileNotFoundError("data/retail.duckdb is missing")
                    # Read-only avoids fabricating an empty DB when injection fails.
                    with duckdb.connect(str(DB), read_only=True) as conn:
                        return conn.execute(query).fetchone()

                def table_exists():
                    count = sql("SELECT count(*) FROM information_schema.tables WHERE table_schema='main' AND table_name='sales_dirty'")[0]
                    return count == 1, f"matching tables={count}"

                def row_count():
                    count = sql("SELECT count(*) FROM sales_dirty")[0]
                    return count > 100000, f"rows={count}; required >100000"

                check("sales_dirty exists", table_exists)
                check("sales_dirty row count", row_count)
                for issue, query in INJECTION_QUERIES.items():
                    with log.open("a", encoding="utf-8") as stream:
                        stream.write(f"\n-- Targeted check: {issue}\n{query};\n")

                    def injection_check(query=query, issue=issue):
                        row = sql(query)
                        count = row[0] if row else 0
                        with log.open("a", encoding="utf-8") as stream:
                            stream.write(f"-- {issue}: detectable rows={count}\n")
                        minimum = 5000 if issue == "I3_schema_drift_price" else 1
                        return count >= minimum, f"detectable rows={count}"

                    check(issue, injection_check)
            elif step == "run-agent":
                def health_check():
                    import pandas as pd
                    import duckdb
                    frame = pd.read_parquet(ROOT / "outputs/data_health.parquet")
                    with duckdb.connect(str(DB), read_only=True) as conn:
                        expected = {row[0] for row in conn.execute("DESCRIBE sales_dirty").fetchall()}
                    actual = set(frame.column_name)
                    return len(frame) == len(expected) and actual == expected and frame.column_name.is_unique, f"profile rows={len(frame)}; table columns={len(expected)}"

                def proposal_check():
                    import pandas as pd
                    frame = pd.read_parquet(ROOT / "outputs/proposals.parquet")
                    return len(frame) >= 5, f"proposals={len(frame)}; required >=5"

                def nonnull_check(column):
                    import pandas as pd
                    frame = pd.read_parquet(ROOT / "outputs/proposals.parquet")
                    count = int(frame[column].isna().sum())
                    return count == 0 and not frame.empty, f"null {column} values={count}"

                check("Profile row per column", health_check)
                check("At least 5 proposals", proposal_check)
                check("Non-null action", lambda: nonnull_check("action"))
                check("Non-null rationale", lambda: nonnull_check("rationale"))
                for name in ("data_health", "proposals", "injected_ground_truth", "summary"):
                    def bundle_check(name=name):
                        import pandas as pd
                        frame = pd.read_parquet(ROOT / f"outputs/powerbi/{name}.parquet")
                        return not frame.empty, f"rows={len(frame)}"
                    check(f"Power BI {name}", bundle_check)
            elif step == "eval":
                report = ROOT / "eval/report.md"
                check("Evaluation report exists", lambda: (report.is_file(), "eval/report.md"))

                def issue_table():
                    text = report.read_text(encoding="utf-8")
                    header = "| Injected issue | Detected | Action correct | Critic passed | Proposed / expected action |"
                    rows = re.findall(r"^\| (I[1-7]_[a-z_]+) \|", text, re.MULTILINE)
                    expected = set(INJECTION_QUERIES)
                    return header in text and len(rows) == 7 and set(rows) == expected, f"per-issue rows={len(rows)}; required seven unique IDs"

                check("Seven-row issue table", issue_table)
                for metric in ("Recall", "Action precision"):
                    def metric_check(metric=metric):
                        text = report.read_text(encoding="utf-8")
                        match = re.search(r"^\| " + re.escape(metric) + r" \| ([0-9]+(?:\.[0-9]+)?)% \(", text, re.MULTILINE)
                        value = float(match.group(1)) if match else float("nan")
                        return math.isfinite(value) and 0 <= value <= 100, f"{metric}={value}%"
                    check(f"Numeric {metric}", metric_check)
    table = Table(title="Clean-state pipeline verification")
    table.add_column("Check", no_wrap=True)
    table.add_column("Result", no_wrap=True)
    table.add_column("Details", overflow="fold")
    for name, ok, detail in results:
        table.add_row(name, "PASS" if ok else "FAIL", detail)
    console.print(table)
    failures = [name for name, ok, _ in results if not ok]
    console.print(f"Passed: {len(results)-len(failures)}/{len(results)} checks. Exit code: {1 if failures else 0}")
    if failures:
        console.print("Failure summary:")
        for step in ("setup", "inject", "run-agent", "eval"):
            log = logs / f"verify_{step}.log"
            if log.exists() and any(name == f"make {step}" and not ok for name, ok, _ in results):
                lines = log.read_text(encoding="utf-8").splitlines()
                causes = [line.strip() for line in lines if line.startswith(("RuntimeError:", "OpenAIError:", "FileNotFoundError:", "CatalogException:", "duckdb.", "_duckdb.", "error:", "ModuleNotFoundError:", "openai."))]
                console.print(f"- {step}: {causes[-1] if causes else 'Command failed; inspect ' + str(log.relative_to(ROOT))}", markup=False)
        console.print("Failed checks: " + ", ".join(failures), markup=False)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
