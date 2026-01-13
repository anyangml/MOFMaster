"""
MOF-Scientist Backend - TypedDict State Definition
"""

from typing import TypedDict, List, Dict, Any, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Shared state passed between all agents in the workflow.

    This TypedDict serves as the memory for the entire system,
    tracking the query, plan, execution progress, and results.
    """

    # Standard LangChain chat history
    messages: Annotated[list[AnyMessage], add_messages]

    # The original goal parsed from user input
    original_query: str

    # The structured plan of execution (list of tool names)
    plan: List[str]

    # Current execution status
    current_step: int

    # Shared dictionary for tool outputs (keys = step names)
    tool_outputs: Dict[str, Any]

    # Supervisor feedback
    review_feedback: str
    is_plan_approved: bool
