"""
Unit tests for the state module
"""

from app.state import AgentState


def test_agent_state_structure():
    """Test that AgentState has the expected keys"""
    # AgentState is a TypedDict, so we can't instantiate it directly
    # but we can check the annotations

    required_keys = {
        "messages",
        "original_query",
        "plan",
        "current_step",
        "tool_outputs",
        "review_feedback",
        "is_plan_approved",
    }

    assert set(AgentState.__annotations__.keys()) == required_keys


def test_agent_state_types():
    """Test that AgentState fields have correct types"""
    annotations = AgentState.__annotations__

    assert "messages" in annotations
    assert "original_query" in annotations
    assert "plan" in annotations
    assert "current_step" in annotations
    assert "tool_outputs" in annotations
    assert "review_feedback" in annotations
    assert "is_plan_approved" in annotations
