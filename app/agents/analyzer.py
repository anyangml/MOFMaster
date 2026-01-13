"""
Analyzer Agent - Scoping, Context Gathering, and Planning
"""

from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.state import AgentState
from app.utils.llm import get_analyzer_llm


def load_knowledge_base() -> str:
    """Load the README_KNOWLEDGE.md file"""
    kb_path = Path(__file__).parent.parent.parent / "README_KNOWLEDGE.md"
    with open(kb_path, "r") as f:
        return f.read()


ANALYZER_SYSTEM_PROMPT = """You are a computational chemistry assistant specializing in Metal-Organic Frameworks (MOFs).

Your job is to:
1. Check if the user query is IN SCOPE (refer to the knowledge base below)
2. Check if you have all necessary CONTEXT to proceed
3. Generate a STEP-BY-STEP PLAN using available tools

KNOWLEDGE BASE:
{knowledge_base}

INSTRUCTIONS:
- If the query is OUT OF SCOPE, politely explain what you cannot do and suggest alternatives
- If you're missing context (e.g., user asks for energy but no structure provided), ask for it
- If ready to proceed, output a plan as a JSON list of tool names

OUTPUT FORMAT when ready to plan:
```json
{{
  "status": "ready",
  "plan": ["tool_name_1", "tool_name_2", "tool_name_3"]
}}
```

OUTPUT FORMAT when need more info:
```json
{{
  "status": "need_context",
  "question": "What information do you need from the user?"
}}
```

OUTPUT FORMAT when out of scope:
```json
{{
  "status": "out_of_scope",
  "reason": "Why this is not supported"
}}
```

Available tool names:
- search_mof_db
- optimize_structure_ase
- calculate_energy_force
"""


def analyzer_node(state: AgentState) -> AgentState:
    """
    Analyzer Agent - First node in the graph.

    Reads user input, validates scope, checks for necessary context,
    and generates a plan if ready.
    """

    # Load knowledge base
    knowledge_base = load_knowledge_base()

    # Get the latest user message
    messages = state["messages"]
    if not messages:
        return state

    # Get user query
    user_query = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
            user_query = msg.content
            break

    if not user_query:
        return state

    # Create prompt
    llm = get_analyzer_llm()

    system_message = SystemMessage(
        content=ANALYZER_SYSTEM_PROMPT.format(knowledge_base=knowledge_base)
    )

    # Invoke LLM
    response = llm.invoke([system_message] + messages)

    # Parse response - look for JSON in the content
    content = response.content

    # Try to extract JSON from markdown code blocks
    import json
    import re

    json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))

            if parsed.get("status") == "ready" and "plan" in parsed:
                # We have a valid plan
                state["plan"] = parsed["plan"]
                state["original_query"] = user_query
                state["current_step"] = 0
                state["messages"].append(
                    AIMessage(
                        content=f"I've created a plan to address your request: {', '.join(parsed['plan'])}"
                    )
                )
            elif parsed.get("status") == "need_context":
                # Need more information
                state["messages"].append(
                    AIMessage(content=parsed.get("question", "I need more information."))
                )
            elif parsed.get("status") == "out_of_scope":
                # Out of scope
                state["messages"].append(
                    AIMessage(
                        content=f"I'm sorry, but this request is outside my current capabilities. {parsed.get('reason', '')}"
                    )
                )
        except json.JSONDecodeError:
            # If JSON parsing fails, just add the response as-is
            state["messages"].append(response)
    else:
        # No JSON found, add response as-is
        state["messages"].append(response)

    return state
