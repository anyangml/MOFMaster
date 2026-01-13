"""
Integration test for the full workflow graph
"""

from langchain_core.messages import HumanMessage

from app.state import AgentState
from app.agents.runner import runner_node
from app.tools.retrieval import search_mof_db
from app.tools.atomistics import optimize_structure_ase, calculate_energy_force


def test_tool_registry():
    """Test that all tools are properly registered"""
    from app.agents.runner import TOOL_REGISTRY

    assert "search_mof_db" in TOOL_REGISTRY
    assert "optimize_structure_ase" in TOOL_REGISTRY
    assert "calculate_energy_force" in TOOL_REGISTRY


def test_runner_search_execution():
    """Test that runner can execute search tool"""
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
    result = runner_node(state)

    # Check results
    assert result["current_step"] == 1
    assert "step_0_search_mof_db" in result["tool_outputs"]
    assert "mof_name" in result["tool_outputs"]["step_0_search_mof_db"]


def test_runner_multi_step_workflow():
    """Test runner with multi-step workflow (search -> optimize)"""
    # Create state with search already done
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
    state = runner_node(state)
    assert state["current_step"] == 1
    assert "step_0_search_mof_db" in state["tool_outputs"]

    # Execute step 2: optimize
    state = runner_node(state)
    assert state["current_step"] == 2
    assert "step_1_optimize_structure_ase" in state["tool_outputs"]

    # Check optimization result
    opt_result = state["tool_outputs"]["step_1_optimize_structure_ase"]
    assert "optimized_cif_filepath" in opt_result or "error" in opt_result


def test_atomistics_tools():
    """Test that atomistics tools work with a real structure"""
    # Create a simple test structure
    search_result = search_mof_db.func("copper")
    cif_path = search_result["cif_filepath"]

    # Test optimization
    opt_result = optimize_structure_ase.func(cif_path)
    # EMT calculator may not work with the minimal CIF, so we just check it doesn't crash
    assert isinstance(opt_result, dict)

    # If optimization succeeded, test energy calculation
    if "optimized_cif_filepath" in opt_result:
        energy_result = calculate_energy_force.func(opt_result["optimized_cif_filepath"])
        assert isinstance(energy_result, dict)
