"""
Example usage of the MOF-Scientist Backend

This script demonstrates how to use the MOF-Scientist workflow graph
without needing to run the full API server.

Note: This example does not require API keys since it only tests
the Runner and tool execution, not the LLM-powered agents.
"""

from langchain_core.messages import HumanMessage

from app.state import AgentState
from app.agents.runner import runner_node


def example_search_workflow():
    """Example: Search for a MOF"""
    print("\n" + "=" * 60)
    print("Example 1: Search for a copper-based MOF")
    print("=" * 60)

    # Create initial state
    state: AgentState = {
        "messages": [HumanMessage(content="copper")],
        "original_query": "copper",
        "plan": ["search_mof_db"],
        "current_step": 0,
        "tool_outputs": {},
        "review_feedback": "",
        "is_plan_approved": True,
    }

    # Execute the plan
    state = runner_node(state)

    # Print results
    print("\nResults:")
    for key, value in state["tool_outputs"].items():
        print(f"\n{key}:")
        for k, v in value.items():
            print(f"  {k}: {v}")


def example_optimization_workflow():
    """Example: Search and optimize a MOF"""
    print("\n" + "=" * 60)
    print("Example 2: Search for a MOF and optimize its structure")
    print("=" * 60)

    # Create initial state
    state: AgentState = {
        "messages": [HumanMessage(content="zinc MOF")],
        "original_query": "zinc",
        "plan": ["search_mof_db", "optimize_structure_ase"],
        "current_step": 0,
        "tool_outputs": {},
        "review_feedback": "",
        "is_plan_approved": True,
    }

    # Execute step 1: Search
    print("\nStep 1: Searching for MOF...")
    state = runner_node(state)
    print(f"  Found: {state['tool_outputs']['step_0_search_mof_db'].get('mof_name', 'N/A')}")

    # Execute step 2: Optimize
    print("\nStep 2: Optimizing structure...")
    state = runner_node(state)

    opt_result = state["tool_outputs"]["step_1_optimize_structure_ase"]
    if "error" in opt_result:
        print(f"  Optimization failed: {opt_result['error']}")
    else:
        print("  Optimization complete:")
        print(f"    Initial energy: {opt_result.get('initial_energy_ev', 'N/A'):.2f} eV")
        print(f"    Final energy: {opt_result.get('final_energy_ev', 'N/A'):.2f} eV")
        print(f"    Energy change: {opt_result.get('energy_change_ev', 'N/A'):.2f} eV")
        print(f"    Steps: {opt_result.get('n_steps', 'N/A')}")
        print(f"    Output: {opt_result.get('optimized_cif_filepath', 'N/A')}")


def example_full_workflow():
    """Example: Full workflow - search, optimize, and calculate energy"""
    print("\n" + "=" * 60)
    print("Example 3: Full workflow (search -> optimize -> energy)")
    print("=" * 60)

    # Create initial state
    state: AgentState = {
        "messages": [HumanMessage(content="high surface area MOF")],
        "original_query": "high surface area",
        "plan": [
            "search_mof_db",
            "optimize_structure_ase",
            "calculate_energy_force",
        ],
        "current_step": 0,
        "tool_outputs": {},
        "review_feedback": "",
        "is_plan_approved": True,
    }

    # Execute all steps
    step_names = ["Search", "Optimize", "Calculate Energy"]
    for i, step_name in enumerate(step_names):
        print(f"\nStep {i + 1}: {step_name}...")
        state = runner_node(state)

        # Show result
        result_key = list(state["tool_outputs"].keys())[-1]
        result = state["tool_outputs"][result_key]

        if "error" in result:
            print(f"  Failed: {result['error']}")
            break
        elif "mof_name" in result:
            print(f"  Found: {result['mof_name']}")
        elif "optimized_cif_filepath" in result:
            print(f"  Energy: {result.get('final_energy_ev', 'N/A'):.2f} eV")
        elif "energy_ev" in result:
            print(f"  Energy: {result['energy_ev']:.2f} eV")
            print(f"  Max force: {result['max_force_ev_ang']:.4f} eV/Å")


if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("MOF-Scientist Backend - Example Usage")
    print("=" * 60)

    # Run examples
    example_search_workflow()
    example_optimization_workflow()
    example_full_workflow()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60 + "\n")
