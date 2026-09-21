# Evaluation Report

| Metric | Result |
| --- | --- |
| Recall | 100.0% (7/7) |
| Action precision | 57.1% (4/7) |

| Injected issue | Detected | Action correct | Critic passed | Proposed / expected action |
| --- | --- | --- | --- | --- |
| I1_null_units | yes | no | yes | investigate / impute |
| I2_duplicate_rows | yes | no | no | investigate / drop |
| I3_schema_drift_price | yes | yes | no | schema_fix / schema_fix |
| I4_out_of_range_price | yes | yes | yes | investigate / investigate |
| I5_negative_units | yes | yes | yes | investigate / investigate |
| I6_timezone_shift | yes | no | yes | investigate / normalize |
| I7_category_typos | yes | yes | yes | normalize / normalize |

Missed injected issues: none in this run.

## Method

Detection includes all candidates, even critic-rejected ones. Critic passed is assessed on the same representative proposal as action correctness.
Matches require table + column + rule signal family. Duplicates use __table__; multi-column issues match any listed column.
Representative selection: critic-approved first, highest confidence next, then anomaly ID. Expected actions never influence selection.
Actions match the manifest enum exactly (or one explicit pipe-delimited alternative); normalize/schema_fix are not silently counted as impute/investigate.
Recall denominator is all manifest entries, including zero-affected injections. Action precision denominator is detected issues; no detections yields N/A.
These are issue-level metrics, not row-level localization or finding precision. Additional false positives do not reduce action precision.
A family match is evidence of a rule firing, not proof of injection causality. Inherent clean-data defects can produce the same signal.
Unknown anomaly prefixes (including unresolved_tz) do not count as detections.

## Explicit Family Maps

```json
{
  "anomaly_prefix": {
    "null_rate": "missingness",
    "duplicates": "duplication",
    "schema_drift": "schema_drift",
    "price_outlier": "price_outlier",
    "negative_units": "negative_units",
    "mixed_tz": "mixed_timezone",
    "category_variants": "category_typos"
  },
  "ground_truth_id": {
    "I1_null_units": "missingness",
    "I2_duplicate_rows": "duplication",
    "I3_schema_drift_price": "schema_drift",
    "I4_out_of_range_price": "price_outlier",
    "I5_negative_units": "negative_units",
    "I6_timezone_shift": "mixed_timezone",
    "I7_category_typos": "category_typos"
  }
}
```

## Provenance

Generated UTC: 2026-09-19T21:37:53.741359+00:00
Manifest seed: 42
Candidate rows: 10; run IDs: ['fb7e11f4-f682-4fa8-933c-721600e889eb']
- `outputs/proposals.parquet` SHA-256: `8370c96ce655d1c1dc02f7c11a41f2eaf23d1a4c6c3b7547eddf667a6cb6df0e`
- `data/ground_truth.json` SHA-256: `9f4a7a235a3390c116edde09953cdc144d3613e1780a41f4a475e4a9e719f430`
