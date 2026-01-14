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
    # Create initial state - use "copper" as query which should match HKUST-1
    state: AgentState = {
        "messages": [HumanMessage(content="copper")],
        "original_query": "copper",
        "plan": ["search_mof_db"],
        "current_step": 0,
        "tool_outputs": {},
        "review_feedback": "",
        "is_plan_approved": True,
    }

    # Execute runner
    result = await runner_node(state)

    # Check results
    assert result["current_step"] == 1
    assert "step_0_search_mof_db" in result["tool_outputs"]
    output = result["tool_outputs"]["step_0_search_mof_db"]
    assert "mof_name" in output or "error" in output
    if "mof_name" in output:
        assert output["mof_name"] == "HKUST-1"


@pytest.mark.asyncio
async def test_runner_multi_step_workflow():
    """Test runner with multi-step workflow via MCP (search -> optimize)"""
    # Create state
    state: AgentState = {
        "messages": [HumanMessage(content="Find and optimize a copper MOF")],
        "original_query": "Find and optimize a copper MOF",
        "plan": ["search_mof_db", "optimize_structure_ase"],
        "current_step": 0,
        "tool_outputs": {},
        "review_feedback": "",
        "is_plan_approved": True,
    }

    # Execute step 1: search
    state = await runner_node(state)
    assert state["current_step"] == 1
    assert "step_0_search_mof_db" in state["tool_outputs"]

    # Execute step 2: optimize
    state = await runner_node(state)
    assert state["current_step"] == 2
    assert "step_1_optimize_structure_ase" in state["tool_outputs"]

    # Check optimization result
    opt_result = state["tool_outputs"]["step_1_optimize_structure_ase"]
    assert "optimized_cif_filepath" in opt_result or "error" in opt_result
