"""
Main LangGraph Definition - The Brain of the System
"""

from typing import Literal
from langgraph.graph import StateGraph, END

from app.state import AgentState
from app.agents.analyzer import analyzer_node
from app.agents.supervisor import supervisor_node
from app.agents.runner import runner_node
from app.agents.reporter import reporter_node


def should_continue_to_supervisor(state: AgentState) -> Literal["supervisor", "end"]:
    """
    Routing function after Analyzer.

    If we have a plan, go to Supervisor for review.
    Otherwise, end (e.g., if out of scope or need more context).
    """
    if state.get("plan") and len(state.get("plan", [])) > 0:
        return "supervisor"
    return "end"


def should_continue_after_supervisor(state: AgentState) -> Literal["runner", "analyzer"]:
    """
    Routing function after Supervisor.

    If plan is approved, go to Runner.
    If rejected, go back to Analyzer for re-planning.
    """
    if state.get("is_plan_approved", False):
        return "runner"
    return "analyzer"


def should_continue_runner(state: AgentState) -> Literal["runner", "reporter"]:
    """
    Routing function after Runner.

    If more steps remain, continue with Runner.
    Otherwise, go to Reporter.
    """
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])

    if current_step < len(plan):
        return "runner"
    return "reporter"


def create_graph() -> StateGraph:
    """
    Create the main workflow graph.

    Flow:
    1. User -> Analyzer (scope & plan)
    2. Analyzer -> Supervisor (review) OR END (if out of scope)
    3. Supervisor -> Runner (if approved) OR Analyzer (if rejected)
    4. Runner -> Runner (loop through steps) OR Reporter (when done)
    5. Reporter -> END
    """

    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("runner", runner_node)
    workflow.add_node("reporter", reporter_node)

    # Set entry point
    workflow.set_entry_point("analyzer")

    # Add edges
    workflow.add_conditional_edges(
        "analyzer", should_continue_to_supervisor, {"supervisor": "supervisor", "end": END}
    )

    workflow.add_conditional_edges(
        "supervisor", should_continue_after_supervisor, {"runner": "runner", "analyzer": "analyzer"}
    )

    workflow.add_conditional_edges(
        "runner", should_continue_runner, {"runner": "runner", "reporter": "reporter"}
    )

    workflow.add_edge("reporter", END)

    return workflow


# Compile the graph
def get_compiled_graph():
    """Get the compiled graph ready for execution"""
    workflow = create_graph()
    return workflow.compile()
