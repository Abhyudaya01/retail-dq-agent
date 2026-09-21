import pytest

from src.agent.nodes import anomaly_flagger_node
from src.agent.state import AgentState


@pytest.mark.parametrize("name,dtype,distinct,values,expected", [
    ("category", "VARCHAR", 4, ["HOBBIES", "Hobbies"], True),
    ("category", "STRING", 49, ["HOBBIES", "Hobbies"], True),
    ("cat_id", "VARCHAR", 4, ["HOBBIES", "Hobbies"], True),
    ("Store_ID", "VARCHAR", 4, ["Store_A", "Store_B"], True),
    ("dept_id", "VARCHAR", 4, ["HOBBIES_1", "HOBBIES_2"], False),
    ("item_id", "VARCHAR", 4, ["HOBBIES_1_001", "HOBBIES_1_002"], False),
    ("category", "VARCHAR", 5, ["A_1", "B_2", "C_3", "Hobbies", "HOBBIES"], False),
    ("category", "VARCHAR", 5, ["A_1", "B_2", "Hobbies", "HOBBIES", "FOODS"], True),
    ("category", "VARCHAR", 4, ["HOBBIES", "HOBBIES "], True),
    ("DATE", "VARCHAR", 4, ["Monday", "Mondai"], False),
    ("timestamp", "VARCHAR", 4, ["Monday", "Mondai"], False),
    ("datetime", "VARCHAR", 4, ["Monday", "Mondai"], False),
    ("category", "VARCHAR", 50, ["HOBBIES", "Hobbies"], False),
    ("category", "INTEGER", 4, ["HOBBIES", "Hobbies"], False),
    ("region", "VARCHAR", 4, ["CA_1", "CA_2"], False),
    ("category", "VARCHAR", 0, [], False),
])
def test_category_guard(name, dtype, distinct, values, expected):
    state = AgentState(profile={"row_count": 100, "duplicate_row_count": 0,
        "columns": {name: {"dtype": dtype, "distinct_count": distinct,
                           "null_rate": .03,
                           "top_values": [{"value": value, "count": 10} for value in values]}}})
    anomaly_flagger_node(state)
    variants = [a for a in state.anomalies if a.id.startswith("category_variants:")]
    assert bool(variants) is expected
    assert all(a.severity == "low" for a in variants)
    assert any(a.id == f"null_rate:{name}" for a in state.anomalies)
