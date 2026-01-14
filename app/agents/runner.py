"""
Runner Agent - Deterministic Tool Execution via MCP
"""

import os
import sys
import asyncio
from typing import Dict, Any, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.state import AgentState

# Configuration for MCP server connection
# This can be configured via environment variables to point to a completely separate repo
MCP_SERVER_COMMAND = os.getenv("MCP_SERVER_COMMAND", sys.executable)
# If provided as a list string like '["-m", "mcp_server.main"]', parse it
MCP_SERVER_ARGS_RAW = os.getenv("MCP_SERVER_ARGS", '["-m", "mcp_server.main"]')
try:
    import json
    MCP_SERVER_ARGS = json.loads(MCP_SERVER_ARGS_RAW)
except:
    MCP_SERVER_ARGS = [MCP_SERVER_ARGS_RAW]

MCP_SERVER_PARAMS = StdioServerParameters(
    command=MCP_SERVER_COMMAND,
    args=MCP_SERVER_ARGS,
    env=os.environ.copy(),
)


async def runner_node(state: AgentState) -> AgentState:
    """
    Runner Agent - Executes tools via MCP server.

    Takes the current step from the plan, calls the MCP server tool,
    and stores the result.
    """

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    tool_outputs = state.get("tool_outputs", {})

    # Check if we're done
    if current_step >= len(plan):
        return state

    # Get the current tool to execute
    tool_name = plan[current_step]

    try:
        # Determine arguments based on tool and previous outputs
        kwargs = _prepare_tool_args(tool_name, tool_outputs, state)

        # Execute via MCP
        async with stdio_client(MCP_SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize session
                await session.initialize()
                
                # Call the tool
                # MCP tools in FastMCP usually have the same name as the function
                result = await session.call_tool(tool_name, kwargs)
                
                # result.content is a list of content objects (TextContent, etc.)
                # We expect a JSON-like dictionary from our implementation
                if result.is_error:
                    tool_outputs[f"step_{current_step}_{tool_name}"] = {
                        "error": str(result.content),
                        "tool_name": tool_name
                    }
                else:
                    # In FastMCP, return values are often wrapped or returned as text
                    # If we returned a dict in the server, it might be in result.content[0].text as JSON
                    # But the MCP SDK often handles this mapping. 
                    # Assuming the content is what we returned.
                    output_data = result.content[0].text if result.content else {}
                    
                    # Try to parse if it's a string, otherwise use as is
                    import json
                    try:
                        if isinstance(output_data, str):
                            output_data = json.loads(output_data)
                    except:
                        pass
                        
                    tool_outputs[f"step_{current_step}_{tool_name}"] = output_data

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
    Prepare arguments for tool execution.
    """

    if tool_name == "search_mof_db":
        return {"query_string": state.get("original_query", "")}

    elif tool_name == "optimize_structure_ase":
        cif_filepath = _find_cif_filepath(tool_outputs)
        if not cif_filepath:
            raise ValueError("No CIF file found in previous outputs for optimization")
        return {"cif_filepath": cif_filepath}

    elif tool_name == "calculate_energy_force":
        cif_filepath = _find_cif_filepath(tool_outputs, prefer_optimized=True)
        if not cif_filepath:
            raise ValueError("No CIF file found in previous outputs for energy calculation")
        return {"cif_filepath": cif_filepath}

    else:
        return {}


def _find_cif_filepath(tool_outputs: Dict[str, Any], prefer_optimized: bool = False) -> str:
    """
    Find a CIF filepath in the tool outputs.
    """

    optimized_path = None
    original_path = None

    # Search through outputs in order
    for key in sorted(tool_outputs.keys()):
        output = tool_outputs[key]

        if isinstance(output, dict):
            if "optimized_cif_filepath" in output:
                optimized_path = output["optimized_cif_filepath"]

            if "cif_filepath" in output and not output.get("error"):
                original_path = output["cif_filepath"]

    if prefer_optimized and optimized_path:
        return optimized_path

    return optimized_path or original_path
