"""
Runner Agent - Deterministic Tool Execution
"""

from typing import Dict, Any, Callable

from app.state import AgentState
from app.tools.retrieval import search_mof_db
from app.tools.atomistics import optimize_structure_ase, calculate_energy_force


# Map tool names to actual functions
TOOL_REGISTRY: Dict[str, Callable] = {
    "search_mof_db": search_mof_db.func,  # Get the underlying function from langchain tool
    "optimize_structure_ase": optimize_structure_ase.func,
    "calculate_energy_force": calculate_energy_force.func,
}


def runner_node(state: AgentState) -> AgentState:
    """
    Runner Agent - Executes tools deterministically without LLM.

    Takes the current step from the plan, maps it to a Python function,
    executes it, and stores the result. This is a pure execution engine
    with no reasoning or decision-making.
    """

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    tool_outputs = state.get("tool_outputs", {})

    # Check if we're done
    if current_step >= len(plan):
        return state

    # Get the current tool to execute
    tool_name = plan[current_step]

    if tool_name not in TOOL_REGISTRY:
        # Unknown tool
        tool_outputs[f"step_{current_step}_{tool_name}"] = {"error": f"Unknown tool: {tool_name}"}
        state["tool_outputs"] = tool_outputs
        state["current_step"] = current_step + 1
        return state

    # Execute the tool
    tool_func = TOOL_REGISTRY[tool_name]

    try:
        # Determine arguments based on tool and previous outputs
        kwargs = _prepare_tool_args(tool_name, tool_outputs, state)

        # Execute
        result = tool_func(**kwargs)

        # Store result
        tool_outputs[f"step_{current_step}_{tool_name}"] = result

    except Exception as e:
        # Store error
        tool_outputs[f"step_{current_step}_{tool_name}"] = {"error": str(e), "tool_name": tool_name}

    # Update state
    state["tool_outputs"] = tool_outputs
    state["current_step"] = current_step + 1

    return state


def _prepare_tool_args(
    tool_name: str, tool_outputs: Dict[str, Any], state: AgentState
) -> Dict[str, Any]:
    """
    Prepare arguments for tool execution based on tool name and previous outputs.

    This implements simple data flow logic:
    - search_mof_db: Extract query from original_query
    - optimize_structure_ase: Get cif_filepath from search results or previous optimization
    - calculate_energy_force: Get cif_filepath from optimization or search
    """

    if tool_name == "search_mof_db":
        # Extract query from the original user query
        # For simplicity, use the whole query
        return {"query_string": state.get("original_query", "")}

    elif tool_name == "optimize_structure_ase":
        # Need a CIF filepath - look in previous outputs
        cif_filepath = _find_cif_filepath(tool_outputs)
        if not cif_filepath:
            raise ValueError("No CIF file found in previous outputs for optimization")
        return {"cif_filepath": cif_filepath}

    elif tool_name == "calculate_energy_force":
        # Prefer optimized structure, fallback to original
        cif_filepath = _find_cif_filepath(tool_outputs, prefer_optimized=True)
        if not cif_filepath:
            raise ValueError("No CIF file found in previous outputs for energy calculation")
        return {"cif_filepath": cif_filepath}

    else:
        return {}


def _find_cif_filepath(tool_outputs: Dict[str, Any], prefer_optimized: bool = False) -> str:
    """
    Find a CIF filepath in the tool outputs.

    Args:
        tool_outputs: Dictionary of previous tool outputs
        prefer_optimized: If True, prefer optimized structures over original

    Returns:
        CIF filepath or None
    """

    optimized_path = None
    original_path = None

    # Search through outputs in order
    for key in sorted(tool_outputs.keys()):
        output = tool_outputs[key]

        if isinstance(output, dict):
            # Check for optimized structure
            if "optimized_cif_filepath" in output:
                optimized_path = output["optimized_cif_filepath"]

            # Check for original structure
            if "cif_filepath" in output and not output.get("error"):
                original_path = output["cif_filepath"]

    # Return based on preference
    if prefer_optimized and optimized_path:
        return optimized_path

    return optimized_path or original_path
