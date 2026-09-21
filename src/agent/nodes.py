from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

from src.agent.llm import structured_llm

from src.agent.prompts import SYSTEM_PROMPT, example_messages
from src.agent.state import AgentState, Anomaly, Proposal
from src.db import get_conn, quote_identifier


def profiler_node(state: AgentState, db_path=None):
    """Compute all statistics in DuckDB, including unresolved date heuristics."""
    table = quote_identifier(state.table_name)
    with get_conn(db_path or os.getenv("RETAIL_DQ_DB", "data/retail.duckdb")) as conn:
        schema = conn.execute(f"DESCRIBE {table}").fetchall()
        row_count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        unique = conn.execute(f"SELECT count(*) FROM (SELECT DISTINCT * FROM {table})").fetchone()[0]
        profile = {"row_count": row_count, "duplicate_row_count": row_count - unique,
                   "schema_hash": hashlib.sha256(json.dumps([r[0] for r in schema]).encode()).hexdigest(),
                   "columns": {}}
        for name, dtype, *_ in schema:
            col = quote_identifier(name)
            numeric = dtype.startswith(("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "FLOAT", "DOUBLE", "DECIMAL"))
            nulls, distinct, minimum, maximum = conn.execute(
                f"SELECT count(*) FILTER (WHERE {col} IS NULL), count(DISTINCT {col}), min({col}), max({col}) FROM {table}"
            ).fetchone()
            stats = {"dtype": dtype, "null_rate": nulls / row_count if row_count else 0.,
                     "distinct_count": distinct, "min": minimum, "max": maximum,
                     "mean": None, "top_values": []}
            if numeric:
                mean, std, negatives = conn.execute(f"SELECT avg({col}), stddev_pop({col}), count(*) FILTER (WHERE {col} < 0) FROM {table}").fetchone()
                stats.update(mean=mean, stddev=std, negative_count=negatives)
                if name in ("sell_price", "price_usd"):
                    count, max_z = conn.execute(f"SELECT count(*) FILTER (WHERE ({col} - ?) / NULLIF(?, 0) > 6), max(({col} - ?) / NULLIF(?, 0)) FROM {table}", [mean, std, mean, std]).fetchone()
                    stats.update(price_outlier_count=count, max_price_z_score=max_z)
            else:
                stats["top_values"] = [{"value": value, "count": count} for value, count in conn.execute(
                    f"SELECT {col}, count(*) AS n FROM {table} WHERE {col} IS NOT NULL GROUP BY {col} ORDER BY n DESC, {col} LIMIT 5").fetchall()]
            dateish = "DATE" in dtype or "TIMESTAMP" in dtype or any(token in name.lower() for token in ("date", "timestamp", "time"))
            if dateish:
                text = f"CAST({col} AS VARCHAR)"
                aware = f"regexp_matches({text}, '(Z|[+-][0-9]{{2}}:?[0-9]{{2}})$', 'i')"
                parsed = f"TRY_CAST({col} AS TIMESTAMPTZ)"
                valid, aware_count, naive_count, date_min, date_max = conn.execute(
                    f"SELECT count({parsed}), count(*) FILTER (WHERE {aware}), count(*) FILTER (WHERE {col} IS NOT NULL AND NOT {aware}), CAST(min({parsed}) AS VARCHAR), CAST(max({parsed}) AS VARCHAR) FROM {table}").fetchone()
                non_null = row_count - nulls
                stats.update(date_min=date_min, date_max=date_max, tz_aware_count=aware_count,
                             tz_naive_count=naive_count, date_parse_failure_count=non_null-valid,
                             tz_check_confidence="low", tz_check_note="Lexical suffix heuristic; offsets and local-day semantics require source confirmation.")
            profile["columns"][name] = stats
    # Keep state JSON serializable, including DuckDB date and Decimal results.
    profile = json.loads(json.dumps(profile, default=str))
    state.profile = profile
    return {"profile": profile}


def anomaly_flagger_node(state: AgentState):
    profile = state.profile
    if profile is None:
        raise ValueError("Run profiler before flagger")
    anomalies = []

    def add(kind, column, signal, value, severity, evidence):
        anomalies.append(Anomaly(id=f"{kind}:{column}", column=column, signal=signal,
                                 value=value, severity=severity, evidence=evidence))

    if profile["duplicate_row_count"]:
        add("duplicates", "__table__", "duplicate rows > 0", profile["duplicate_row_count"], "high", {"row_count": profile["row_count"]})
    columns = profile["columns"]
    for name, stats in columns.items():
        if stats["null_rate"] > .005:
            add("null_rate", name, "null_rate > 0.005", stats["null_rate"], "med", {"row_count": profile["row_count"]})
        if name == "units" and stats.get("negative_count", 0):
            add("negative_units", name, "units < 0", stats["negative_count"], "high", {"min": stats["min"]})
        if stats.get("price_outlier_count", 0):
            add("price_outlier", name, "price z-score > 6", stats["max_price_z_score"], "high", {"count": stats["price_outlier_count"], "mean": stats["mean"], "stddev": stats["stddev"]})
        if "tz_check_confidence" in stats:
            if stats["tz_aware_count"] and stats["tz_naive_count"]:
                add("mixed_tz", name, "mixed timezone suffixes", stats["tz_aware_count"], "med", {key: value for key, value in stats.items() if key.startswith("tz_") or key.startswith("date_")})
            elif stats["date_parse_failure_count"]:
                add("unresolved_tz", name, "timezone check unresolved for unparseable dates", stats["date_parse_failure_count"], "low", {"confidence": "low", "note": stats["tz_check_note"]})
        pairs = []
        values = [entry["value"] for entry in stats["top_values"] if isinstance(entry["value"], str)]
        systematic_identifiers = bool(values) and (
            all(re.fullmatch(r"[A-Z]+_\d+", value) for value in values)
            or sum("_" in value and any(char.isdigit() for char in value) for value in values) / len(values) >= .6
        )
        eligible = (
            stats["dtype"].upper() in {"VARCHAR", "STRING"}
            and stats["distinct_count"] < 50
            and name.lower() not in {"date", "timestamp", "datetime"}
            and bool(values)
            and not systematic_identifiers
        )
        if eligible:
            for left, right in combinations(values, 2):
                distance = _levenshtein(left.casefold(), right.casefold())
                if distance <= 2:
                    pairs.append({"left": left, "right": right, "distance": distance})
        if pairs:
            add("category_variants", name, "Levenshtein <= 2 on top categorical values", len(pairs), "low", {"pairs": pairs})
    if "sell_price" in columns and "price_usd" in columns:
        add("schema_drift", "sell_price", "both sell_price and price_usd present", True, "high", {"columns": ["sell_price", "price_usd"]})
    state.anomalies = anomalies
    return {"anomalies": anomalies}


def _levenshtein(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (a != b)))
        previous = current
    return previous[-1]


def get_proposal_llm():
    _require_api_key()
    return structured_llm(Proposal, temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")))


def _require_api_key():
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")


def fix_proposer_node(state: AgentState):
    if not state.anomalies:
        return {"proposals": []}
    _require_api_key()
    llm = get_proposal_llm()
    groups = defaultdict(list)
    for anomaly in state.anomalies:
        groups[anomaly.column].append(anomaly)
    proposals = []
    traces = dict(state.llm_traces)
    for column, anomalies in groups.items():
        context = {"table_name": state.table_name, "column": column,
                   "profile": state.profile["columns"].get(column, {"row_count": state.profile["row_count"]})}
        # One request per anomaly, grouped by column with a compact shared prefix.
        prefix = [("system", SYSTEM_PROMPT), *example_messages(), ("human", json.dumps(context))]
        for anomaly in anomalies:
            messages = [*prefix, ("human", anomaly.model_dump_json())]
            proposal = Proposal.model_validate(llm.invoke(messages))
            if proposal.anomaly_id != anomaly.id or not 0 <= proposal.confidence <= 1:
                raise ValueError("LLM proposal has incorrect anomaly ID or confidence")
            proposals.append(proposal)
            traces[anomaly.id] = [{"node": "proposer", "messages": messages,
                                   "output": proposal.model_dump()}]
    state.proposals = proposals
    return {"proposals": proposals, "llm_traces": traces}
