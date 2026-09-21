"""LangGraph agent components for retail data-quality workflows."""

def build_graph(db_path=None, *, use_critic=True):
    """Load graph assembly lazily so the module CLI is not pre-imported."""
    from src.agent.graph import build_graph as assemble

    return assemble(db_path, use_critic=use_critic)

__all__ = ["build_graph"]
