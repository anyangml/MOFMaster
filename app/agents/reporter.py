"""
Reporter Agent - Data Synthesis and Summary Generation
"""

from langchain_core.messages import SystemMessage, AIMessage

from app.state import AgentState
from app.utils.llm import get_reporter_llm


REPORTER_SYSTEM_PROMPT = """You are a scientific reporter synthesizing computational chemistry results for Metal-Organic Framework (MOF) workflows.

Your job is to create a clear, well-formatted Markdown report that:
1. Directly answers the original user query and scientific goals.
2. Summarizes the workflow that was actually executed (tools and order).
3. Presents key numerical results with proper units.
4. Clearly cites all relevant structures and file paths.
5. Interprets results (e.g., stability, trends, comparisons) rather than just listing raw numbers.

FORMATTING REQUIREMENTS:
- Use Markdown headers, lists, and, when helpful, simple tables.
- Always include units (eV for energy, Å for distances, eV/Å for forces).
- Cite file paths for structures used (e.g., CIF files, optimized structures).
- When multiple MOFs are involved, clearly indicate which results correspond to which structure.
- Be concise but complete, and highlight the most important findings for the user.

ORIGINAL QUERY:
{original_query}

EXECUTED PLAN (tool sequence):
{plan}

TOOL OUTPUTS (structured data to base your report on):
{tool_outputs}

Generate a professional, user-facing Markdown report that explains what was done, what was found, and how it relates to the user’s question.
"""


async def reporter_node(state: AgentState) -> AgentState:
    """
    Reporter Agent - Final node that synthesizes results.

    Takes all tool outputs and generates a human-readable
    Markdown report answering the original query.
    """

    original_query = state.get("original_query", "")
    plan = state.get("plan", [])
    tool_outputs = state.get("tool_outputs", {})

    # If no outputs, return early
    if not tool_outputs:
        state["messages"].append(
            AIMessage(content="No results to report - workflow did not complete successfully.")
        )
        return state

    # Create report prompt
    llm = get_reporter_llm()

    # Format tool outputs for readability
    formatted_outputs = _format_tool_outputs(tool_outputs)

    system_message = SystemMessage(
        content=REPORTER_SYSTEM_PROMPT.format(
            original_query=original_query,
            plan="\n".join(f"{i+1}. {step}" for i, step in enumerate(plan)),
            tool_outputs=formatted_outputs,
        )
    )

    # Generate report
    response = await llm.ainvoke([system_message])

    # Add to messages
    state["messages"].append(response)

    return state


def _format_tool_outputs(tool_outputs: dict) -> str:
    """Format tool outputs as readable text"""

    lines = []
    for key, value in tool_outputs.items():
        lines.append(f"\n### {key}")

        if isinstance(value, dict):
            for k, v in value.items():
                lines.append(f"- {k}: {v}")
        else:
            lines.append(f"- {value}")

    return "\n".join(lines)
