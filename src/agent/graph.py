from functools import partial
from pathlib import Path
from datetime import datetime, timezone
import json
import os
from uuid import uuid4

import typer
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

from src.agent.nodes import anomaly_flagger_node, fix_proposer_node, profiler_node
from src.agent.state import AgentState
from src.io_out import export_powerbi, write_outputs
from src.critic import critic_node
from src.db import get_conn

app = typer.Typer(help="Run the retail data-quality agent with rubric review.")


@app.callback()
def cli():
    pass


def _traced(name, node):
    def invoke(state: AgentState):
        inputs = state.model_dump(mode="json", exclude={"node_traces", "llm_traces"})
        outputs = node(state)
        serialized = json.loads(json.dumps(outputs, default=lambda value: value.model_dump(mode="json")))
        return {**outputs, "node_traces": [*state.node_traces,
                {"node": name, "inputs": inputs,
                 "outputs": {k: v for k, v in serialized.items() if k != "llm_traces"}}]}
    return invoke


def build_graph(db_path=None, *, use_critic=True):
    graph = StateGraph(AgentState)
    graph.add_node("profiler", _traced("profiler", partial(profiler_node, db_path=db_path)))
    graph.add_node("flagger", _traced("flagger", anomaly_flagger_node))
    graph.add_node("proposer", _traced("proposer", fix_proposer_node))
    graph.add_node("critic", _traced("critic", partial(critic_node, enabled=use_critic)))
    graph.add_edge(START, "profiler")
    graph.add_edge("profiler", "flagger")
    graph.add_edge("flagger", "proposer")
    graph.add_edge("proposer", "critic")
    graph.add_edge("critic", END)
    return graph.compile()


def persist_fixes(state, db_path=None):
    """Append a completed run atomically, retaining rejected and bypassed fixes."""
    with get_conn(db_path or os.getenv("RETAIL_DQ_DB", "data/retail.duckdb")) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS fixes (
            run_id VARCHAR, ts TIMESTAMPTZ, anomaly_id VARCHAR, "column" VARCHAR,
            action VARCHAR, rationale VARCHAR, confidence DOUBLE,
            critic_specificity INTEGER, critic_grounding INTEGER, critic_safety INTEGER,
            critic_sql_validity INTEGER, critic_total INTEGER, critic_passes BOOLEAN, trace_json VARCHAR
        )''')
        anomalies = {a.id: a for a in state.anomalies}
        ts = datetime.now(timezone.utc)
        rows = []
        for proposal in state.proposals:
            verdict = state.critic_verdicts.get(proposal.anomaly_id, {})
            scores = verdict.get("scores", {})
            trace = {"run_id": state.run_id, "table_name": state.table_name,
                     "anomaly": anomalies[proposal.anomaly_id].model_dump(),
                     "nodes": state.node_traces, "llm_calls": state.llm_traces.get(proposal.anomaly_id, []),
                     "critic_status": "reviewed" if verdict else "bypassed"}
            rows.append([state.run_id, ts, proposal.anomaly_id, anomalies[proposal.anomaly_id].column,
                         proposal.action, proposal.rationale, proposal.confidence,
                         *[scores.get(k) for k in ("specificity", "grounding", "safety", "sql_validity")],
                         verdict.get("total"), verdict.get("passes"), json.dumps(trace)])
        if rows:
            conn.execute("BEGIN TRANSACTION")
            try:
                conn.executemany("INSERT INTO fixes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise


def run(table_name="sales_dirty", *, db_path=None, use_critic=True) -> AgentState:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    state = AgentState.model_validate(build_graph(db_path, use_critic=use_critic).invoke(
        AgentState(table_name=table_name, run_id=str(uuid4()))))
    persist_fixes(state, db_path)
    return state


@app.command("run")
def run_cli(table_name: str = "sales_dirty", db: Path = Path("data/retail.duckdb"),
            output_dir: Path = Path("outputs"),
            powerbi: bool = typer.Option(False, "--export-powerbi"),
            ground_truth: Path = Path("data/ground_truth.json"),
            no_critic: bool = typer.Option(False, "--no-critic")):
    state = run(table_name, db_path=db, use_critic=not no_critic)
    write_outputs(state, output_dir)
    if powerbi:
        export_powerbi(state, ground_truth, output_dir / "powerbi")
    typer.echo(f"Wrote {len(state.surfaced_proposals)} surfaced and {len(state.rejected_proposals)} rejected proposals to {output_dir}; run_id={state.run_id}")


if __name__ == "__main__":
    app()
