from unittest.mock import Mock

import pytest

from src.agent.nodes import fix_proposer_node
from src.agent.state import AgentState, Anomaly, Proposal
from src.critic import critique


def test_proposer_and_critic_fail_before_client_construction(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    proposer_factory, critic_factory = Mock(), Mock()
    monkeypatch.setattr("src.agent.nodes.get_proposal_llm", proposer_factory)
    monkeypatch.setattr("src.critic.get_critic_llm", critic_factory)
    state = AgentState(anomalies=[Anomaly(id="null_rate:units", column="units",
                                         signal="null_rate > 0.005", value=.03,
                                         severity="med", evidence={})])
    message = "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
    with pytest.raises(RuntimeError) as error:
        fix_proposer_node(state)
    assert str(error.value) == message
    with pytest.raises(RuntimeError) as error:
        critique(Proposal(anomaly_id="null_rate:units", action="investigate",
                          rationale="Missing daily sales", sql_or_pseudocode="Review units", confidence=.8), {})
    assert str(error.value) == message
    proposer_factory.assert_not_called()
    critic_factory.assert_not_called()
