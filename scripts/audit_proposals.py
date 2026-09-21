#!/usr/bin/env -S uv run python
"""Print proposal samples and literal grounding signals for manual inspection."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
DOMAIN_PATTERN = re.compile(
    r"\b(?:sales?|pric(?:e[sd]?|ing)|shel(?:f|ves)|inventor(?:y|ies)|"
    r"retail(?:ers?|ing)?|units?|skus?|stores?|dates?|dating)\b", re.IGNORECASE
)


def grounding_checks(rationale: str, column: str) -> tuple[bool, bool]:
    return (bool(DOMAIN_PATTERN.search(rationale)),
            bool(re.search(r"(?<!\w)" + re.escape(column) + r"(?!\w)", rationale, re.IGNORECASE)))


def sample_proposals(frame: pd.DataFrame, worst: bool = False) -> pd.DataFrame:
    count = min(5, len(frame))
    if worst:
        return frame.sort_values(["confidence", "anomaly_id"], kind="stable").head(count)
    shuffled = frame.sample(frac=1, random_state=42).reset_index(drop=True)
    # Keep one path per reachable coverage state. Once a dimension reaches three,
    # its exact members no longer matter. This finds feasible coverage, not a guess.
    states = {(0, frozenset(), frozenset()): ()}

    def extend(values, value):
        if values is None:
            return None
        expanded = values | {value}
        return None if len(expanded) >= 3 else expanded

    for position, row in shuffled.iterrows():
        additions = {}
        for (size, columns, actions), chosen in states.items():
            if size < count:
                key = (size + 1, extend(columns, row.column_name), extend(actions, row.action))
                if key not in states:
                    additions.setdefault(key, (*chosen, position))
        states.update(additions)

    def coverage(key):
        _, columns, actions = key
        c = 3 if columns is None else len(columns)
        a = 3 if actions is None else len(actions)
        return (int(c >= 3) + int(a >= 3), min(c, a), c + a, c, a)

    candidates = [key for key in states if key[0] == count]
    best = max(candidates, key=coverage)
    return shuffled.iloc[list(states[best])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worst", action="store_true", help="Show the five lowest-confidence candidates.")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/proposals.parquet")
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    required = {"anomaly_id", "column_name", "signal", "action", "rationale", "confidence"}
    if not required <= set(frame):
        parser.error(f"Missing fields: {sorted(required - set(frame))}")
    if frame.empty:
        parser.error("No proposals to sample")
    if frame[list(required)].isna().any().any():
        parser.error("Required proposal fields contain nulls")
    sample = sample_proposals(frame, args.worst)
    console = Console(width=100, force_terminal=False)
    console.print(f"Mode: {'worst (lowest confidence)' if args.worst else 'sample (seed=42)'}; "
                  f"rows={len(sample)}; columns={sample.column_name.nunique()}; actions={sample.action.nunique()}")
    domain_hits = column_hits = 0
    for _, row in sample.iterrows():
        domain, column = grounding_checks(row.rationale, row.column_name)
        domain_hits += domain
        column_hits += column
        content = Text(f"anomaly_id: {row.anomaly_id}\ncolumn: {row.column_name}\n"
                       f"signal: {row.signal}\naction: {row.action}\n"
                       f"rationale: {row.rationale}\nconfidence: {row.confidence}")
        console.print(Panel(content))
        console.print(f"domain_terms_hit={domain} | column_name_hit={column}")
    console.print(f"Grounding heuristic: {domain_hits}/{len(sample)} hit domain terms, "
                  f"{column_hits}/{len(sample)} reference the column by name.")


if __name__ == "__main__":
    main()
