"""Regenerate the real pipeline with a fixed configuration for a Loom demo."""
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from rich.console import Console

from eval.eval import evaluate_files
from src.agent.graph import run
from src.injection import main as inject
from src.io_out import export_powerbi, write_outputs

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
console = Console()


def main():
    load_dotenv(ROOT / ".env")
    raw = ROOT / "data/raw"
    for name in ("sales_train_evaluation.csv", "sell_prices.csv", "calendar.csv"):
        if not (raw / name).is_file():
            raise SystemExit(f"Missing raw input: {raw / name}. Drop all three M5 CSVs there first.")
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        raise SystemExit("Set OPENAI_API_KEY in .env before the live demo.")
    os.environ.update(OPENAI_MODEL="gpt-4o-mini-2024-07-18", OPENAI_TEMPERATURE="0", OPENAI_SEED="42")
    db, truth = ROOT / "data/retail.duckdb", ROOT / "data/ground_truth.json"
    console.print("60-second narration: 0-10s raw slice; 10-20s seven seeded defects; "
                  "20-35s rules and proposals; 35-45s critic and traces; 45-60s results and report bundle.")
    console.print("Live regeneration follows. Stage durations depend on disk and API latency.")
    started = perf_counter()
    inject(db=db, raw_dir=raw, ground_truth=truth, seed=42, load=True)
    console.print(f"Loaded and injected: {perf_counter()-started:.1f}s elapsed")
    state = run(db_path=db)
    console.print(f"Agent reviewed {len(state.proposals)} candidates: {perf_counter()-started:.1f}s elapsed")
    write_outputs(state, ROOT / "outputs")
    export_powerbi(state, truth, ROOT / "outputs/powerbi")
    evaluate_files(ROOT / "outputs/proposals.parquet", truth, ROOT / "eval/report.md", ROOT / "README.md")
    console.print(f"Complete: {perf_counter()-started:.1f}s; run_id={state.run_id}")
    console.print("Open eval/report.md and follow powerbi/model_notes.md for the manual report.")


if __name__ == "__main__":
    main()
