# Manual Power BI Report Build

Allow about 30 minutes once the four exports exist. This guide builds an Import
model for one profiled table, normally `sales_dirty`.

## Generate the Bundle

Run injection first if needed, configure `.env`, then run:

```bash
make inject
uv run python -m src.agent.graph run --export-powerbi
```

The second command runs the agent (including the LLM proposer) once and writes
both ordinary outputs and these four files under `outputs/powerbi/`:

- `data_health.parquet`
- `proposals.parquet`
- `injected_ground_truth.parquet`
- `summary.parquet`

For a completed in-memory state, exporting needs no further model calls:

```python
from src.io_out import export_powerbi

export_powerbi(state)
```

The manifest must exist at `data/ground_truth.json`; missing inputs raise an error
rather than creating fabricated ground truth. Export uses the current state and
manifest, so regenerate both from the same data slice and seed.

## Import and Types (0-8 Minutes)

In Power BI Desktop choose **Get Data -> Parquet**, browse to each local file,
then choose **Transform Data**. Name the queries exactly as shown below, without
file extensions. Repeat for all four files, check types, then **Close & Apply**.
Power BI Desktop must be able to access these paths; if exports are generated on
a Mac, transfer all four files to the Windows machine running Desktop.

### data_health

Grain: one row per column of the profiled table.

| Columns | Power BI Type | Use |
| --- | --- | --- |
| table_name, column_name, dtype, schema_hash | Text | Identification and schema |
| row_count, duplicate_row_count, distinct_count | Whole number | Counts |
| null_rate, mean | Decimal number | Null rate is a 0-1 fraction |
| min, max | Text | Mixed numeric/text extrema stored as text |
| top_values | Text | JSON evidence, not a list/struct |
| stddev | Decimal number | Numeric columns only |
| negative_count, price_outlier_count | Whole number | Numeric evidence |
| max_price_z_score | Decimal number | Price evidence |
| date_min, date_max | Text | Parsed timestamp extrema |
| tz_aware_count, tz_naive_count, date_parse_failure_count | Whole number | Date evidence |
| tz_check_confidence, tz_check_note | Text | Heuristic limitations |

Optional evidence columns appear only when the profiled schema supports them;
other rows are blank. Do not convert `min`/`max` to numeric globally. Format
`null_rate` as a percentage. `row_count` and `duplicate_row_count` repeat for every
column: use MAX, never SUM. Null rate excludes no rows: it is nulls / table rows.

### proposals

Grain: one proposal per detected anomaly. Anomaly details are already joined in;
no separate `anomalies` table is exported.

| Columns | Power BI Type |
| --- | --- |
| table_name, anomaly_id, column_name, signal, severity | Text |
| action, rationale, sql_or_pseudocode | Text |
| value, evidence_json, match_key | Text |
| confidence | Decimal number, formatted as percentage |

Severity values are `low`, `med`, `high`. Actions are `drop`, `impute`,
`investigate`, `schema_fix`, `normalize`. `__table__` identifies duplicate-row
findings. JSON evidence and SQL are review text; neither is executed by the report.

### injected_ground_truth

Grain: one injected issue-column pair. Multi-column issues repeat `issue_id`;
count distinct issue IDs, not rows. Duplicate-row injection is represented once
using `__table__`, matching the table-level anomaly.

| Columns | Power BI Type |
| --- | --- |
| issue_id, table_name, column_name, match_key | Text |
| detection_signal, recommended_action, rationale | Text |
| affected_row_estimate | Whole number |

Affected counts describe source rows changed before duplication. They repeat for
multi-column issues; do not sum them across issue-column rows. Issues with zero
affected rows remain in the manifest.

### summary

Grain: exactly one row for the completed run; keep this table disconnected.

| Columns | Power BI Type |
| --- | --- |
| total_issues, high_severity_count, proposals_generated | Whole number |
| avg_confidence | Decimal number, percentage (surfaced candidates only) |

`total_issues` counts detected anomalies, not the seven injection types.
`avg_confidence` is blank when no candidates surface. Summary cards describe the
whole run and do not respond to proposal slicers.

## Relationships (8-11 Minutes)

Disable/delete automatically detected relationships and add only:

- `data_health[column_name]` (1) -> `proposals[column_name]` (*), single direction
  from data_health. Leave `__table__` proposals unmatched; avoid filtering these
  out by column when assessing whole-table duplicates. This is valid for this
  single-table export; use table+column keys if combining runs/tables later.
- Keep `injected_ground_truth` disconnected for the recommended model. Both it
  and proposals have `match_key = table_name | column_name`. The coverage measure
  below applies a best-effort virtual join with TREATAS, avoiding an ambiguous
  many-to-many physical relationship.

For a side-by-side audit table, optionally create a Power Query reference of
`proposals`, left-merge `injected_ground_truth` on `match_key`, and expand
`issue_id` and `detection_signal`. Disable load on that query unless using the
audit visual. The merge can multiply proposal rows: never compute KPIs from it.
Column overlap alone is not verified detection: a null finding on units can
match both null-unit and negative-unit injections. Coverage below is therefore
column coverage, not precision or recall.

If adding a separate anomalies table later, relate its unique `id` (1) to
`proposals[anomaly_id]` (*), single direction. It is unnecessary for this bundle.

## Six Measures (11-16 Minutes)

Create these six measures verbatim (Modeling -> New measure). Put them in
`summary` for organization; their home table does not change filtering.

```dax
Total Issues =
COALESCE(MAX('summary'[total_issues]), 0)
```

```dax
Row Count =
COALESCE(MAX('data_health'[row_count]), 0)
```

```dax
Null Rate =
AVERAGE('data_health'[null_rate])
```

```dax
High Sev % =
DIVIDE(
    CALCULATE(
        DISTINCTCOUNT('proposals'[anomaly_id]),
        KEEPFILTERS('proposals'[severity] = "high")
    ),
    DISTINCTCOUNT('proposals'[anomaly_id]),
    0
)
```

```dax
Avg Confidence =
AVERAGE('proposals'[confidence])
```

```dax
Coverage vs Ground Truth =
VAR ProposalColumns = VALUES('proposals'[match_key])
VAR MatchedIssues =
    CALCULATE(
        DISTINCTCOUNT('injected_ground_truth'[issue_id]),
        REMOVEFILTERS('injected_ground_truth'),
        TREATAS(ProposalColumns, 'injected_ground_truth'[match_key])
    )
VAR InjectedIssues =
    CALCULATE(
        DISTINCTCOUNT('injected_ground_truth'[issue_id]),
        REMOVEFILTERS('injected_ground_truth')
    )
RETURN
    DIVIDE(COALESCE(MatchedIssues, 0), InjectedIssues, 0)
```

Format Null Rate, High Sev %, Avg Confidence, and Coverage vs Ground Truth as
percentages with one decimal place. Null Rate is the average across visible
columns, equivalently the share of missing cells when every column has the same
row count. Coverage respects proposal/column slicers and counts a multi-column
issue covered when at least one of its columns matches. Its denominator includes
all manifest issue IDs, even zero-affected issues.

## Tab 1: Data Health Report (16-23 Minutes)

Use a 16:9 page. Keep the title at the top, four KPI cards in one row underneath,
then a matrix on the left and horizontal bar chart on the right.

| Visual | Field Configuration |
| --- | --- |
| Row count card | `[Row Count]` |
| Null rate card | `[Null Rate]` |
| Duplicate count card | `data_health[duplicate_row_count]`, aggregation MAX |
| High-severity count card | `summary[high_severity_count]`, aggregation MAX |
| Column x metric matrix | Rows: `data_health[column_name]`; Values: `[Null Rate]`, MAX distinct_count, MAX mean, MAX negative_count (if present) |
| Null-rate bar chart | Y: `data_health[column_name]`; X: MAX null_rate; sort descending |

For the matrix, turn off row subtotals/grand totals for counts, format null rate
as a percentage, and apply conditional background color to null rate (red above
0.5%). Use blanks for inapplicable numeric metrics. Add dtype, min, max, and
top_values as chart tooltips rather than trying to aggregate JSON/text in a
numeric matrix. Do not apply proposal severity slicers to profile statistics.

## Tab 2: Proposed Fixes (23-28 Minutes)

Place severity and action slicers across the top, an average-confidence card at
the top right, and a full-width table beneath.

| Visual | Field Configuration |
| --- | --- |
| Severity slicer | `proposals[severity]` |
| Action slicer | `proposals[action]` |
| Average-confidence card | `[Avg Confidence]` |
| Fixes table | `proposals[column_name]`, signal, severity, action, rationale, confidence |

Rename `column_name` to "Column" for this visual. Set confidence to Don't
summarize, show one decimal percentage, enable rationale word wrap, and widen
rationale to about half the table. Include anomaly_id as a tooltip/extra audit
column if identical visible rows are collapsed by the visual. Use red/amber/gray
severity colors. Optional small cards use `[Total Issues]`, `[High Sev %]`, and
`[Coverage vs Ground Truth]`; label coverage "Best-effort column coverage".

## Refresh and Check (28-30 Minutes)

1. Generate the current bundle before refreshing. `make all` runs setup,
   injection, the reviewed agent, Power BI export, and issue-family evaluation.
   `make eval` writes `eval/report.md` and the README results marker block.
2. In Desktop use **Home -> Refresh**. Each query should point to its specific
   file via Get Data -> Parquet. Keep filenames fixed; update Source paths in
   Transform Data when moving the bundle to another machine.
3. Check that summary has one row, data_health has one row per profiled column,
   and proposals has one row per anomaly_id. Ground truth normally has seven
   distinct issue IDs, with more than seven rows due to multi-column issues.
4. Check duplicate/row cards use MAX, confidence lies in 0-1, and summary issue
   count equals the distinct proposal anomaly count for a completed agent run.
5. Clear slicers before saving. Local-file refresh in the Power BI Service needs
   a configured gateway with access to these paths; Desktop refresh is sufficient
   for this manual build.

Connector and virtual-join references:
[Microsoft Parquet connector](https://learn.microsoft.com/en-us/power-query/connectors/parquet)
and [Microsoft TREATAS reference](https://learn.microsoft.com/en-us/dax/treatas-function-dax).

## Critic Review and Show Rejected Toggle

The exporter retains every candidate. Additional proposal columns:

| Columns | Power BI Type |
| --- | --- |
| run_id, proposal_status, critic_notes, rejection_reason | Text |
| is_rejected, critic_passes | True/False (critic_passes can be blank) |
| critic_specificity, critic_grounding, critic_safety, critic_sql_validity, critic_total | Whole number (nullable) |

Statuses are `surfaced` (critic total >= 14/20), `rejected`, `bypassed` (explicit
`--no-critic`), and `pending` (exported candidate not yet reviewed). Display status,
critic_total, critic_notes, and rejection_reason in the Tab 2 table. Rejecting a
candidate does not delete it from DuckDB or Parquet. Grade 5 is best on every
dimension, including safety. SQL validity is an LLM judgment, not executed SQL.

For a quick toggle that also filters cards, add a slicer on
`proposals[is_rejected]`, rename it "Show rejected", and save with False selected.
False shows non-rejected candidates; clearing the slicer includes rejected ones;
True shows only rejected ones. Apply a page filter on `proposal_status` to include
only surfaced, bypassed, and rejected, excluding pending. Bypassed candidates
remain explicitly labeled and have blank critic scores.

For literal Hide/Show buttons instead, keep that slicer hidden in the Selection
pane and create two bookmarks capturing **Data** for **Selected visuals** (only
the rejection slicer): "Hide rejected" selects False; "Show rejected" clears
the selection. Add a bookmark navigator and save on Hide rejected. Capturing only
that slicer preserves the user's severity and action selections.

The existing Avg Confidence and High Sev % measures follow the current slicers;
best-effort Coverage follows visible proposals too. Total Issues and summary
cards remain whole-run totals. Update the Tab 1 consistency check to compare
summary totals against **all** candidate anomaly IDs, not only surfaced ones.
