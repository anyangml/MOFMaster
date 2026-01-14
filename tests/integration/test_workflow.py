"""
Integration test for the full workflow graph
"""

import pytest
from langchain_core.messages import HumanMessage

from app.state import AgentState
from app.agents.runner import runner_node


@pytest.mark.asyncio
async def test_runner_search_execution():
    """Test that runner can execute search tool via MCP"""
    # Create initial state
    state: AgentState = {
        "messages": [HumanMessage(content="HKUST-1")],
        "original_query": "HKUST-1",
        "plan": ["search_mofs"],
        "current_step": 0,
        "tool_outputs": {},
        "review_feedback": "",
        "is_plan_approved": True,
    }

    # Execute runner
    result = await runner_node(state)

    # Check results
    assert result["current_step"] == 1
    assert "step_0_search_mofs" in result["tool_outputs"]
    output = result["tool_outputs"]["step_0_search_mofs"]
    assert "HKUST-1" in str(output)


@pytest.mark.asyncio
async def test_runner_multi_step_workflow():
    """Test runner with multi-step workflow via MCP (search -> optimize)"""
    # Create state
    state: AgentState = {
        "messages": [HumanMessage(content="Find and optimize HKUST-1")],
        "original_query": "HKUST-1",
        "plan": ["search_mofs", "optimize_structure"],
        "current_step": 0,
        "tool_outputs": {},
        "review_feedback": "",
        "is_plan_approved": True,
    }

    # Execute step 1: search
    state = await runner_node(state)
    assert state["current_step"] == 1
    assert "step_0_search_mofs" in state["tool_outputs"]

    # Execute step 2: optimize
    state = await runner_node(state)
    assert state["current_step"] == 2
    assert "step_1_optimize_structure" in state["tool_outputs"]

    # Check optimization result
    opt_result = state["tool_outputs"]["step_1_optimize_structure"]
    assert "Successfully" in str(opt_result)
