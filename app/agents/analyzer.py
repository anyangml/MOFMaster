import logging
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.state import AgentState
from app.utils.llm import get_analyzer_llm

# Setup logger
logger = logging.getLogger(__name__)


def load_knowledge_base() -> str:
    """Load the README_KNOWLEDGE.md file"""
    kb_path = Path(__file__).parent.parent.parent / "README_KNOWLEDGE.md"
    with open(kb_path, "r") as f:
        return f.read()


ANALYZER_SYSTEM_PROMPT = """You are the Planning & Analysis agent for the MOF-Scientist backend.

You specialize in computational chemistry workflows for Metal-Organic Frameworks (MOFs).

Your responsibilities are to:
1. Determine whether the user query is IN SCOPE (using the knowledge base below).
2. Check if you have all necessary CONTEXT to proceed (structures, CIF paths, etc.).
3. Design a SCIENTIFICALLY SOUND PLAN using the available tools. Use single-step plans for simple tasks and multi-step plans when necessary.
4. Use the knowledge base as your contract for how tools should be combined.
5. Avoid unnecessary computation, but do include optimization/energy steps when they are scientifically appropriate for the user’s goal.

KNOWLEDGE BASE (authoritative description of role, tools, and workflows):
{knowledge_base}

{feedback_section}

PLANNING GUIDELINES:
- For simple tasks, a single-step plan may be sufficient. For complex tasks, think in terms of multi-step workflows.
- Follow the default order of operations when appropriate:
    - structure acquisition → geometry optimization → energy/force calculation.
- It is acceptable to:
    - Use only `search_mofs` when the user only wants candidates or a quick lookup.
    - Use `optimize_structure` → `calculate_energy` when the user provides a specific structure.
    - Perform screening workflows over multiple candidates (e.g., search → filter → optimize/energy for a small subset).
- Do NOT add expensive steps (especially energy calculations) if the user explicitly requested to avoid them.
- If the user’s intent is ambiguous (e.g., "find a stable Cu-based MOF"), you may include optimization and/or energy calculations as part of a reasonable scientific workflow.
- If there is supervisor feedback, carefully consider it and improve your plan accordingly.

SCOPE AND CONTEXT HANDLING:
- If the query is OUT OF SCOPE according to the knowledge base, politely explain what you cannot do and, when possible, suggest alternative analyses you *can* perform.
- If you are missing critical context (e.g., user asks for energy but no structure or CIF path is available and cannot be inferred), ask a concise clarification question.

OUTPUT REQUIREMENTS:
- When you are ready to plan, you MUST output a **valid JSON object** in one of the formats below. Do not include any extra text outside the JSON.

OUTPUT FORMAT when ready to plan:
```json
{{
    "status": "ready",
    "plan": ["tool_name_1", "tool_name_2"]
}}
```

The `plan` is an ordered list of tool names, each entry being one of the available tools below. Tools may appear multiple times if needed.

OUTPUT FORMAT when you need more information:
```json
{{
    "status": "need_context",
    "question": "What information do you need from the user?"
}}
```

OUTPUT FORMAT when the request is out of scope:
```json
{{
    "status": "out_of_scope",
    "reason": "Why this is not supported"
}}
```

Available tool names (must match exactly):
- search_mofs
- optimize_structure
- calculate_energy
"""


async def analyzer_node(state: AgentState) -> AgentState:
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

    # Check if there's supervisor feedback (from a previous rejection)
    review_feedback = state.get("review_feedback", "")
    rejection_count = state.get("_rejection_count", 0)
    
    # Build feedback section if there's feedback
    if review_feedback and rejection_count > 0:
        feedback_section = f"""
PREVIOUS PLAN REJECTION FEEDBACK (attempt {rejection_count}):
The supervisor rejected the previous plan with this feedback:
{review_feedback}

IMPORTANT: Please carefully consider this feedback and create an improved plan that addresses the supervisor's concerns.
"""
    else:
        feedback_section = ""

    # Create prompt
    llm = get_analyzer_llm()

    system_message = SystemMessage(
        content=ANALYZER_SYSTEM_PROMPT.format(
            knowledge_base=knowledge_base,
            feedback_section=feedback_section
        )
    )

    # Invoke LLM
    try:
        response = await llm.ainvoke([system_message] + messages)
    except Exception as e:
        # Avoid surfacing provider errors as 500s. Return a structured JSON response per contract.
        logger.exception("❌ Analyzer: LLM invocation failed")
        import json

        msg = str(e)
        if "filtered due to the prompt triggering" in msg.lower() or "content management policy" in msg.lower():
            payload = {
                "status": "out_of_scope",
                "reason": "The upstream LLM provider blocked this request due to content filtering. Please rephrase and try again.",
            }
        else:
            payload = {
                "status": "out_of_scope",
                "reason": f"The planning model failed unexpectedly: {msg}",
            }

        return {
            "messages": [AIMessage(content=json.dumps(payload, ensure_ascii=False))],
            "original_query": user_query or "",
            "plan": [],
            "current_step": 0,
            "is_plan_approved": False,
        }

    # Parse response - look for JSON in the content
    content = response.content
    logger.debug(f"🔍 Analyzer: Raw LLM response content (first 500 chars): {content[:500]}")

    # Try to extract JSON from the content
    import json
    import re

    parsed = None
    
    # Strategy 1: Look for JSON in markdown code blocks
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            logger.info(f"✅ Analyzer: Found and parsed JSON from markdown block")
        except json.JSONDecodeError:
            logger.warning(f"⚠️  Analyzer: Found markdown block but JSON parsing failed")

    # Strategy 2: If no valid JSON yet, look for any curly brace structure
    if not parsed:
        # Try to find the first '{' and the last '}'
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = content[start:end+1]
            try:
                parsed = json.loads(json_str)
                logger.info(f"✅ Analyzer: Found and parsed JSON using brace matching")
            except json.JSONDecodeError:
                logger.warning(f"⚠️  Analyzer: Found brace structure but JSON parsing failed")

    # Strategy 3: Try parsing the entire content
    if not parsed:
        try:
            parsed = json.loads(content.strip())
            logger.info(f"✅ Analyzer: Parsed entire response as JSON")
        except json.JSONDecodeError:
            pass

    if parsed and isinstance(parsed, dict):
        logger.debug(f"DEBUG Analyzer parsed JSON: {json.dumps(parsed)}")
        try:
            if parsed.get("status") == "ready" and "plan" in parsed:
                logger.info(f"✅ Analyzer: Successfully processed 'ready' status")
                
                # Store previous plan before creating new one (for supervisor comparison)
                current_plan = state.get("plan", [])
                
                # We have a valid plan - extract it explicitly
                new_plan_raw = parsed["plan"]
                
                if not isinstance(new_plan_raw, list):
                    new_plan = list(new_plan_raw) if new_plan_raw else []
                else:
                    new_plan = new_plan_raw.copy()
                
                # Create message content
                plan_str = ', '.join(new_plan)
                
                # Re-read rejection_count from state
                current_rejection_count = state.get("_rejection_count", 0)
                is_revision = current_rejection_count > 0
                
                if is_revision:
                    message_content = f"I've revised the plan based on your feedback (revision #{current_rejection_count}): {plan_str}"
                else:
                    message_content = f"I've created a plan to address your request: {plan_str}"
                
                # Create the message
                new_message = AIMessage(content=message_content)
                
                # Determine what to return
                updates = {
                    "messages": [new_message],
                    "plan": new_plan,
                    "original_query": user_query,
                    "current_step": 0
                }
                
                if current_plan:
                    updates["_previous_plan"] = current_plan
                
                return updates
                
            elif parsed.get("status") == "need_context":
                payload = {
                    "status": "need_context",
                    "question": parsed.get("question", "I need more information."),
                }
                return {"messages": [AIMessage(content=json.dumps(payload, ensure_ascii=False))], "plan": []}
            elif parsed.get("status") == "out_of_scope":
                payload = {"status": "out_of_scope", "reason": parsed.get("reason", "")}
                return {"messages": [AIMessage(content=json.dumps(payload, ensure_ascii=False))], "plan": []}
        except Exception as e:
            logger.error(f"❌ Analyzer: Error processing parsed JSON: {e}")
            return {"messages": [response]}
    
    # If we get here, no valid JSON was found or it couldn't be processed
    logger.warning(f"⚠️  Analyzer: No valid JSON plan found in response")
    return {"messages": [response]}
