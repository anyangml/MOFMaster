"""
Runner Agent - Deterministic Tool Execution via MCP (HTTP)
"""

import os
import asyncio
import json
from typing import Dict, Any, List
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.state import AgentState

# Configuration for MCP server connection
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp")


async def runner_node(state: AgentState) -> AgentState:
    """
    Runner Agent - Executes tools via MCP server over Streamable HTTP.
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

        # Execute via Streamable HTTP
        async with streamable_http_client(MCP_SERVER_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, kwargs)
                tool_outputs[f"step_{current_step}_{tool_name}"] = _process_mcp_result(result, tool_name)

    except Exception as e:
        # Store error
        tool_outputs[f"step_{current_step}_{tool_name}"] = {"error": str(e), "tool_name": tool_name}

    # Update state
    state["tool_outputs"] = tool_outputs
    state["current_step"] = current_step + 1

    return state


def _process_mcp_result(result: Any, tool_name: str) -> Dict[str, Any]:
    """Helper to process MCP tool results into standard dictionary format."""
    if result.is_error:
        return {
            "error": str(result.content),
            "tool_name": tool_name
        }
    
    output_data = result.content[0].text if result.content else {}
    
    # Try to parse if it's a string, otherwise use as is
    try:
        if isinstance(output_data, str):
            output_data = json.loads(output_data)
    except:
        pass
        
    return output_data


def _prepare_tool_args(
    tool_name: str, tool_outputs: Dict[str, Any], state: AgentState
) -> Dict[str, Any]:
    """
    Prepare arguments for tool execution.
    """

    if tool_name == "search_mofs":
        return {"query": state.get("original_query", "")}

    elif tool_name == "optimize_structure":
        # Search for a mof name in outputs
        for key, val in tool_outputs.items():
            if isinstance(val, list) and len(val) > 0 and "name" in val[0]:
                return {"name": val[0]["name"]}
            if isinstance(val, dict) and "name" in val:
                return {"name": val["name"]}
        return {"name": "Unknown MOF"}

    elif tool_name == "calculate_energy":
        # Needs data (CIF string or path)
        cif_filepath = _find_cif_filepath(tool_outputs, prefer_optimized=True)
        if not cif_filepath:
             return {"data": "No structure provided"}
        return {"data": cif_filepath}

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
