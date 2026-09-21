from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Anomaly(BaseModel):
    id: str
    column: str
    signal: str
    value: Any
    severity: Literal["low", "med", "high"]
    evidence: dict


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    anomaly_id: str
    action: Literal["drop", "impute", "investigate", "schema_fix", "normalize"]
    rationale: str
    sql_or_pseudocode: str
    confidence: float


class AgentState(BaseModel):
    table_name: str = "sales_dirty"
    profile: dict | None = None
    anomalies: list[Anomaly] = Field(default_factory=list)
    proposals: list[Proposal] = Field(default_factory=list)
    surfaced_proposals: list[Proposal] = Field(default_factory=list)
    rejected_proposals: list[RejectedProposal] = Field(default_factory=list)
    critic_verdicts: dict[str, dict] = Field(default_factory=dict)
    node_traces: list[dict] = Field(default_factory=list)
    llm_traces: dict[str, list[dict]] = Field(default_factory=dict)
    run_id: str | None = None


class RejectedProposal(BaseModel):
    proposal: Proposal
    reason: str


AgentState.model_rebuild()
