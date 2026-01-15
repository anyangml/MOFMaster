# MOFMaster Architecture Deep Dive

## Table of Contents
1. [System Overview](#system-overview)
2. [Design Philosophy](#design-philosophy)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Agent Deep Dive](#agent-deep-dive)
6. [Tool System](#tool-system)
7. [State Management](#state-management)
8. [LLM Integration](#llm-integration)
9. [Error Handling](#error-handling)
10. [Extension Points](#extension-points)

---

## System Overview

MOFMaster is a **multi-agent scientific workflow system** that bridges natural language and computational chemistry. It's built on a **state machine architecture** using LangGraph, where each node represents an agent with specific responsibilities.

### Key Design Principles
1. **Separation of Concerns**: Planning (LLM) vs Execution (deterministic)
2. **Scientific Rigor**: Plans are validated before execution
3. **Data Flow Transparency**: All intermediate results are stored
4. **Extensibility**: New tools and agents can be added easily

### The Big Picture
```
User Query (Natural Language)
    ↓
Analyzer (LLM) → Generates a plan
    ↓
Supervisor (LLM) → Validates the plan
    ↓
Runner (Deterministic) → Executes tools step-by-step
    ↓
Reporter (LLM) → Synthesizes results
    ↓
Response (Markdown Report)
```

---

## Design Philosophy

### Why Multi-Agent?

**Modularity**: Each agent has a single, well-defined responsibility:
- **Analyzer**: "What should we do?"
- **Supervisor**: "Is this plan safe and correct?"
- **Runner**: "Execute the plan exactly as specified"
- **Reporter**: "Explain the results to the user"

This separation allows:
- Independent testing of each component
- Easier debugging (you know which agent failed)
- Flexible modification (swap LLM models per agent)

### Why LangGraph?

LangGraph provides:
- **State Management**: Shared memory across agents
- **Conditional Routing**: Dynamic workflow based on results
- **Checkpointing**: Can save/resume workflows
- **Streaming**: Real-time progress updates

### Why Deterministic Runner?

**Reliability**: LLMs can hallucinate or make mistakes. By separating planning from execution:
- Plans are validated by the Supervisor before execution
- Execution is deterministic (no LLM involved)
- Results are reproducible
- Tool calls are guaranteed to be correct

---

## Core Components

### 1. State (`app/state.py`)

The `AgentState` is the **shared memory** for the entire workflow. Think of it as a blackboard where all agents read and write.

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # Chat history
    original_query: str                                   # User's original question
    plan: List[str]                                       # List of tool names to execute
    current_step: int                                     # Which step we're on
    tool_outputs: Dict[str, Any]                          # Results from each tool
    review_feedback: str                                  # Supervisor's feedback
    is_plan_approved: bool                                # Supervisor's decision
```

**Key Features**:
- **`messages`**: Uses LangChain's `add_messages` annotation for automatic message list management
- **`tool_outputs`**: Keyed by `step_{index}_{tool_name}` for easy data flow tracking
- **`current_step`**: Allows the Runner to loop through the plan

### 2. Graph (`app/graph.py`)

The graph defines the **control flow** of the system.

#### Nodes
- `analyzer`: Entry point, creates a plan
- `supervisor`: Reviews the plan
- `runner`: Executes one step at a time
- `reporter`: Final node, generates report

#### Edges (Decision Points)

**After Analyzer**:
```python
def should_continue_to_supervisor(state: AgentState):
    if state.get("plan") and len(state.get("plan", [])) > 0:
        return "supervisor"  # We have a plan, review it
    return "end"             # No plan (out of scope or needs more info)
```

**After Supervisor**:
```python
def should_continue_after_supervisor(state: AgentState):
    if state.get("is_plan_approved", False):
        return "runner"      # Plan approved, execute it
    return "analyzer"        # Plan rejected, re-plan
```

**After Runner**:
```python
def should_continue_runner(state: AgentState):
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])
    
    if current_step < len(plan):
        return "runner"      # More steps to execute
    return "reporter"        # Done, generate report
```

**Why this matters**: The Runner loops back to itself until all steps are done. This allows multi-step workflows without complex logic.

### 3. API Server (`app/server.py`)

Uses **FastAPI** + **LangServe** to expose the graph as a REST API.

```python
graph = get_compiled_graph()

add_routes(
    app,
    graph,
    path="/mof-scientist",
    enabled_endpoints=["invoke", "stream", "playground"],
)
```

**LangServe Features**:
- **`/invoke`**: Blocking, returns final result
- **`/stream`**: Streams events as they happen
- **`/playground`**: Interactive web UI for testing

---

## Data Flow

Let's trace a real query through the system:

### Example Query: "Find a copper-based MOF and calculate its energy"

#### Step 1: Analyzer
**Input**: `{"messages": [{"role": "user", "content": "Find a copper-based MOF and calculate its energy"}]}`

**Process**:
1. Loads `README_KNOWLEDGE.md` to understand capabilities
2. Constructs a prompt with the knowledge base + user query
3. Invokes LLM (GPT-4o)
4. LLM returns JSON:
   ```json
   {
     "status": "ready",
     "plan": ["search_mof_db", "optimize_structure_ase", "calculate_energy_force"]
   }
   ```
5. Parses JSON and updates state:
   - `state["plan"] = ["search_mof_db", "optimize_structure_ase", "calculate_energy_force"]`
   - `state["current_step"] = 0`
   - `state["original_query"] = "Find a copper-based MOF and calculate its energy"`

**Output**: State with a plan

#### Step 2: Supervisor
**Input**: State with plan

**Process**:
1. Creates a review prompt with the plan
2. Uses **structured output** (Pydantic) to get a `SupervisorReview`:
   ```python
   class SupervisorReview(BaseModel):
       approved: bool
       feedback: str
   ```
3. LLM validates the plan:
   - ✅ Structure acquisition before optimization
   - ✅ Optimization before energy calculation
   - ✅ All tools are available
4. Returns:
   ```json
   {
     "approved": true,
     "feedback": "Plan follows best practices. Structure search before optimization, and optimization before energy calculation."
   }
   ```
5. Updates state:
   - `state["is_plan_approved"] = True`
   - `state["review_feedback"] = "..."`

**Output**: State with approval

#### Step 3: Runner (Iteration 1)
**Input**: State with approved plan, `current_step = 0`

**Process**:
1. Gets tool name: `plan[0] = "search_mof_db"`
2. Looks up tool in registry: `TOOL_REGISTRY["search_mof_db"]`
3. Prepares arguments using `_prepare_tool_args()`:
   ```python
   kwargs = {"query_string": "Find a copper-based MOF and calculate its energy"}
   ```
4. Executes: `result = search_mof_db.func(**kwargs)`
5. Result:
   ```python
   {
     "mof_name": "HKUST-1",
     "formula": "Cu3(BTC)2",
     "cif_filepath": "/path/to/data/HKUST-1.cif",
     "properties": {"surface_area_m2g": 1850, ...}
   }
   ```
6. Stores result: `state["tool_outputs"]["step_0_search_mof_db"] = result`
7. Increments: `state["current_step"] = 1`

**Output**: State with search results

#### Step 4: Runner (Iteration 2)
**Input**: State with `current_step = 1`

**Process**:
1. Gets tool name: `plan[1] = "optimize_structure_ase"`
2. Prepares arguments:
   - Calls `_find_cif_filepath(tool_outputs)` to get the CIF file from previous step
   - Returns: `/path/to/data/HKUST-1.cif`
3. Executes: `result = optimize_structure_ase.func(cif_filepath="/path/to/data/HKUST-1.cif")`
4. ASE performs optimization (BFGS + EMT calculator)
5. Result:
   ```python
   {
     "optimized_cif_filepath": "/path/to/data/HKUST-1_optimized.cif",
     "initial_energy_ev": -123.45,
     "final_energy_ev": -125.67,
     "energy_change_ev": -2.22,
     "n_steps": 15,
     "converged": True
   }
   ```
6. Stores: `state["tool_outputs"]["step_1_optimize_structure_ase"] = result`
7. Increments: `state["current_step"] = 2`

**Output**: State with optimization results

#### Step 5: Runner (Iteration 3)
**Input**: State with `current_step = 2`

**Process**:
1. Gets tool name: `plan[2] = "calculate_energy_force"`
2. Prepares arguments:
   - Calls `_find_cif_filepath(tool_outputs, prefer_optimized=True)`
   - Returns optimized CIF: `/path/to/data/HKUST-1_optimized.cif`
3. Executes energy calculation
4. Result:
   ```python
   {
     "energy_ev": -125.67,
     "max_force_ev_ang": 0.03,
     "cif_filepath": "/path/to/data/HKUST-1_optimized.cif",
     "n_atoms": 52
   }
   ```
5. Stores: `state["tool_outputs"]["step_2_calculate_energy_force"] = result`
6. Increments: `state["current_step"] = 3`

**Output**: State with all results

#### Step 6: Reporter
**Input**: State with complete `tool_outputs`

**Process**:
1. Formats tool outputs into a readable structure
2. Creates a prompt:
   ```
   ORIGINAL QUERY: Find a copper-based MOF and calculate its energy
   EXECUTED PLAN: 1. search_mof_db, 2. optimize_structure_ase, 3. calculate_energy_force
   TOOL OUTPUTS: [formatted outputs]
   ```
3. LLM generates a Markdown report:
   ```markdown
   # MOF Analysis Report
   
   ## Structure Found
   - **MOF Name**: HKUST-1
   - **Formula**: Cu3(BTC)2
   - **Structure File**: `/path/to/data/HKUST-1.cif`
   
   ## Optimization Results
   - **Initial Energy**: -123.45 eV
   - **Final Energy**: -125.67 eV
   - **Energy Change**: -2.22 eV
   - **Optimization Steps**: 15
   
   ## Energy Calculation
   - **Total Energy**: -125.67 eV
   - **Maximum Force**: 0.03 eV/Å
   - **Number of Atoms**: 52
   
   The structure has been successfully optimized and is in a low-energy state.
   ```
4. Adds to messages: `state["messages"].append(AIMessage(content=report))`

**Output**: State with final report

#### Step 7: API Response
The `/invoke` endpoint returns:
```json
{
  "output": {
    "content": "# MOF Analysis Report\n...",
    "plan": ["search_mof_db", "optimize_structure_ase", "calculate_energy_force"],
    "tool_outputs": {...}
  }
}
```

---

## Agent Deep Dive

### Analyzer Agent

**Purpose**: Convert natural language → executable plan

**Key Components**:

1. **Knowledge Base Loading**:
   ```python
   def load_knowledge_base() -> str:
       kb_path = Path(__file__).parent.parent.parent / "README_KNOWLEDGE.md"
       with open(kb_path, "r") as f:
           return f.read()
   ```
   The knowledge base is injected into the system prompt, giving the LLM context about:
   - Available tools
   - Valid workflow patterns
   - Scientific rules
   - Scope guidelines

2. **System Prompt**:
   - Instructs the LLM to check scope, context, and generate plans
   - Provides output formats (JSON schemas)
   - Lists available tools

3. **JSON Parsing**:
   ```python
   json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
   ```
   Extracts JSON from markdown code blocks (LLMs often wrap JSON in ```json)

4. **Three Possible Outcomes**:
   - **Ready**: Plan generated, proceed to Supervisor
   - **Need Context**: Missing information, ask user
   - **Out of Scope**: Request not supported

### Supervisor Agent

**Purpose**: Quality control and scientific validation

**Key Features**:

1. **Structured Output**:
   ```python
   class SupervisorReview(BaseModel):
       approved: bool
       feedback: str
   
   structured_llm = llm.with_structured_output(SupervisorReview)
   ```
   Forces the LLM to return a specific structure (no parsing needed!)

2. **Scientific Rules**:
   - Structure acquisition before operations
   - Optimization before energy calculations
   - Tool availability checks

3. **Feedback Loop**:
   - If rejected, state returns to Analyzer
   - Analyzer can re-plan based on feedback

### Runner Agent

**Purpose**: Deterministic tool execution (no LLM)

**Architecture**:

1. **Tool Registry**:
   ```python
   TOOL_REGISTRY: Dict[str, Callable] = {
       "search_mof_db": search_mof_db.func,
       "optimize_structure_ase": optimize_structure_ase.func,
       "calculate_energy_force": calculate_energy_force.func,
   }
   ```
   Maps tool names (strings) to Python functions

2. **Data Flow Logic** (`_prepare_tool_args`):
   - **search_mof_db**: Uses `original_query`
   - **optimize_structure_ase**: Extracts `cif_filepath` from search results
   - **calculate_energy_force**: Prefers optimized CIF, falls back to original

3. **Error Handling**:
   ```python
   try:
       result = tool_func(**kwargs)
       tool_outputs[f"step_{current_step}_{tool_name}"] = result
   except Exception as e:
       tool_outputs[f"step_{current_step}_{tool_name}"] = {"error": str(e), ...}
   ```
   Errors are stored, not raised (workflow continues)

4. **Looping**:
   - Executes one step per invocation
   - Graph loops back to Runner until `current_step >= len(plan)`

### Reporter Agent

**Purpose**: Synthesize and explain results

**Key Features**:

1. **Markdown Generation**:
   - LLM generates formatted reports
   - Includes headers, lists, tables
   - Cites file paths and units

2. **Tool Output Formatting**:
   ```python
   def _format_tool_outputs(tool_outputs: dict) -> str:
       # Converts dict to readable Markdown
   ```

3. **Context Aggregation**:
   - Combines original query + plan + outputs
   - Gives LLM everything it needs to explain

---

## Tool System

### Tool Structure

All tools are **LangChain tools** (decorated with `@tool`):

```python
@tool
def search_mof_db(query_string: str) -> Dict[str, Any]:
    """
    Docstring becomes the tool description.
    LangChain can use this for automatic tool selection.
    """
    ...
```

**Benefits**:
- Automatic type checking
- Docstrings → API documentation
- Compatible with LangChain agents

### Tool Categories

#### 1. Retrieval Tools (`app/tools/retrieval.py`)

**`search_mof_db`**:
- **Input**: Query string (e.g., "copper", "high surface area")
- **Output**: MOF metadata + CIF filepath
- **Implementation**: Simple keyword matching (MVP)
- **Auto-generates CIF**: If CIF doesn't exist, creates a minimal placeholder

**Future Enhancement**: Replace with vector database + embeddings

#### 2. Atomistics Tools (`app/tools/atomistics.py`)

**`optimize_structure_ase`**:
- **Input**: CIF filepath
- **Output**: Optimized CIF + energies + steps
- **Calculator**: EMT (fast, approximate)
- **Optimizer**: BFGS with `fmax=0.05 eV/Å`

**`calculate_energy_force`**:
- **Input**: CIF filepath
- **Output**: Energy (eV) + max force (eV/Å)
- **Static calculation**: No optimization

#### 3. I/O Tools (`app/tools/io.py`)

**`get_data_dir`**:
- Reads `DATA_DIR` from environment
- Creates directory if needed
- Returns `Path` object

**`read_cif_file` / `write_cif_file`**:
- Simple file I/O helpers

### Data Flow Between Tools

The `_find_cif_filepath` function implements **automatic data passing**:

```python
def _find_cif_filepath(tool_outputs, prefer_optimized=False):
    optimized_path = None
    original_path = None
    
    for key in sorted(tool_outputs.keys()):  # Process in order
        output = tool_outputs[key]
        
        if "optimized_cif_filepath" in output:
            optimized_path = output["optimized_cif_filepath"]
        
        if "cif_filepath" in output:
            original_path = output["cif_filepath"]
    
    # Return based on preference
    if prefer_optimized and optimized_path:
        return optimized_path
    return optimized_path or original_path
```

**Why this works**:
- `tool_outputs` keys are sorted: `step_0_...`, `step_1_...`, etc.
- Later steps overwrite earlier paths
- `calculate_energy_force` prefers optimized structures

---

## State Management

### TypedDict vs Pydantic

**Why TypedDict?**:
- LangGraph requires TypedDict for state
- Lightweight (no runtime overhead)
- Type hints for development

**When Pydantic?**:
- API schemas (`app/schema.py`)
- Structured LLM outputs (`SupervisorReview`)

### Message Management

```python
messages: Annotated[list[AnyMessage], add_messages]
```

**What `add_messages` does**:
- Appends new messages to the list
- Handles message deduplication
- Supports various message types (HumanMessage, AIMessage, SystemMessage)

### State Updates

Agents update state **in-place**:

```python
def analyzer_node(state: AgentState) -> AgentState:
    state["plan"] = parsed["plan"]
    state["messages"].append(AIMessage(content="..."))
    return state
```

**Important**: Return the full state, even if only some fields changed

---

## LLM Integration

### Model Factory (`app/utils/llm.py`)

**Purpose**: Centralized LLM configuration

```python
def get_llm(model_name="gpt-4o", temperature=0.0, streaming=False):
    if model_name.startswith("gpt"):
        return ChatOpenAI(model=model_name, ...)
    elif model_name.startswith("claude"):
        return ChatAnthropic(model=model_name, ...)
```

**Benefits**:
- Easy model swapping
- Per-agent configuration
- Supports multiple providers

### Temperature Settings

All agents use `temperature=0.0` (deterministic):
- **Analyzer**: Consistent plans for similar queries
- **Supervisor**: Consistent review criteria
- **Reporter**: Reproducible reports

**Why not higher temperature?**:
- Scientific work requires reproducibility
- Plans must be reliable
- Can increase for creative tasks (future feature)

### Structured Outputs

**Supervisor uses structured outputs**:
```python
structured_llm = llm.with_structured_output(SupervisorReview)
review = structured_llm.invoke([system_message])
# review is a SupervisorReview object, no parsing needed!
```

**Why not use for Analyzer?**:
- Analyzer has 3 different output schemas (ready/need_context/out_of_scope)
- Harder to express in a single Pydantic model
- JSON parsing is flexible enough

---

## Error Handling

### Tool-Level Errors

```python
try:
    result = tool_func(**kwargs)
    tool_outputs[f"step_{current_step}_{tool_name}"] = result
except Exception as e:
    tool_outputs[f"step_{current_step}_{tool_name}"] = {
        "error": str(e),
        "tool_name": tool_name
    }
```

**Strategy**: Catch errors, store them, continue workflow

**Why not raise?**:
- Partial results are still useful
- Reporter can explain what went wrong
- Allows debugging without crashes

### ASE-Specific Errors

ASE tools return errors as dictionaries:
```python
return {"error": f"Optimization failed: {str(e)}", "cif_filepath": cif_filepath}
```

**Reporter handles these gracefully**:
- Checks for `"error"` key in outputs
- Explains what failed and why

### API-Level Errors

**FastAPI handles**:
- Validation errors (Pydantic)
- HTTP errors (404, 500, etc.)

**LangServe adds**:
- Streaming error recovery
- Timeout handling

---

## Extension Points

### Adding New Tools

1. **Create the tool**:
   ```python
   @tool
   def my_new_tool(arg1: str, arg2: int) -> Dict[str, Any]:
       """Tool description for LLM"""
       # Implementation
       return {"result": ...}
   ```

2. **Register it**:
   ```python
   # In app/agents/runner.py
   TOOL_REGISTRY["my_new_tool"] = my_new_tool.func
   ```

3. **Update knowledge base**:
   ```markdown
   # In README_KNOWLEDGE.md
   ### my_new_tool
   **Purpose**: ...
   **Input**: ...
   **Output**: ...
   ```

4. **Add data flow logic** (if needed):
   ```python
   # In _prepare_tool_args
   elif tool_name == "my_new_tool":
       return {"arg1": state["some_field"], "arg2": 42}
   ```

### Adding New Agents

1. **Create agent file**:
   ```python
   # app/agents/my_agent.py
   def my_agent_node(state: AgentState) -> AgentState:
       # Agent logic
       return state
   ```

2. **Add to graph**:
   ```python
   # In app/graph.py
   workflow.add_node("my_agent", my_agent_node)
   workflow.add_edge("some_node", "my_agent")
   ```

3. **Add routing logic** (if conditional):
   ```python
   def should_go_to_my_agent(state):
       if state["some_condition"]:
           return "my_agent"
       return "other_node"
   ```

### Swapping LLM Models

**Per-agent**:
```python
# In app/utils/llm.py
def get_analyzer_llm():
    return get_llm(model_name="claude-3-5-sonnet-20241022", temperature=0.0)
```

**Globally**:
```python
def get_llm(model_name="claude-3-5-sonnet-20241022", ...):
    # All agents now use Claude
```

### Using Different Calculators

**Replace EMT**:
```python
# In app/tools/atomistics.py
from mace.calculators import MACECalculator

atoms.calc = MACECalculator(model_path="path/to/mace/model")
```

**Add calculator selection**:
```python
@tool
def optimize_structure_ase(cif_filepath: str, calculator: str = "emt"):
    if calculator == "emt":
        atoms.calc = EMT()
    elif calculator == "mace":
        atoms.calc = MACECalculator(...)
```

---

## Common Patterns

### 1. Agent Pattern
```python
def agent_node(state: AgentState) -> AgentState:
    # 1. Extract inputs from state
    input_data = state["some_field"]
    
    # 2. Process (LLM or deterministic)
    result = process(input_data)
    
    # 3. Update state
    state["output_field"] = result
    state["messages"].append(AIMessage(content="..."))
    
    # 4. Return state
    return state
```

### 2. Tool Pattern
```python
@tool
def tool_name(arg1: type1, arg2: type2) -> Dict[str, Any]:
    """
    Tool description for LLM.
    
    Args:
        arg1: Description
        arg2: Description
    
    Returns:
        Dictionary with results
    """
    try:
        # Implementation
        return {"key": value}
    except Exception as e:
        return {"error": str(e)}
```

### 3. Conditional Routing Pattern
```python
def routing_function(state: AgentState) -> Literal["option1", "option2"]:
    """
    Decide where to go next based on state.
    """
    if state["some_condition"]:
        return "option1"
    return "option2"
```

---

## Testing Strategy

### Unit Tests
- Test individual tools in isolation
- Test state structure
- Test utility functions

### Integration Tests
- Test multi-step workflows
- Test data flow between tools
- Test Runner with real tools

### Example Test
```python
def test_runner_multi_step_workflow():
    state: AgentState = {
        "plan": ["search_mof_db", "optimize_structure_ase"],
        "current_step": 0,
        "tool_outputs": {},
        ...
    }
    
    # Execute step 1
    state = runner_node(state)
    assert state["current_step"] == 1
    assert "step_0_search_mof_db" in state["tool_outputs"]
    
    # Execute step 2
    state = runner_node(state)
    assert state["current_step"] == 2
    assert "step_1_optimize_structure_ase" in state["tool_outputs"]
```

---

## Performance Considerations

### LLM Latency
- **Analyzer**: ~2-5s (depends on query complexity)
- **Supervisor**: ~1-3s (shorter prompt)
- **Reporter**: ~3-7s (longer generation)

**Total LLM time**: ~6-15s per query

### Tool Execution
- **search_mof_db**: <100ms (in-memory search)
- **optimize_structure_ase**: ~1-10s (depends on structure size)
- **calculate_energy_force**: <1s (static calculation)

### Optimization Ideas
1. **Caching**: Cache search results and optimized structures
2. **Parallel Tools**: Run independent tools in parallel
3. **Streaming**: Stream progress updates to user
4. **Async**: Make tool execution asynchronous

---

## Security Considerations

### API Keys
- Stored in `.env` (not committed)
- Loaded via `python-dotenv`
- Validated at startup

### File I/O
- All CIF files in controlled `DATA_DIR`
- No arbitrary file access
- Path validation needed (future)

### LLM Input Sanitization
- User queries passed directly to LLM
- No SQL injection risk (no database)
- Consider prompt injection protection (future)

---

## Future Enhancements

### Planned Features
1. **Vector Database**: Replace keyword search with semantic search
2. **More Calculators**: MACE, DFT (VASP, Quantum ESPRESSO)
3. **Molecular Dynamics**: Time evolution simulations
4. **Electronic Structure**: Band structure, DOS calculations
5. **Experiment Integration**: Lab automation APIs
6. **Visualization**: 3D structure viewer, plots

### Architecture Improvements
1. **Checkpointing**: Save/resume long workflows
2. **Parallelization**: Run independent tools concurrently
3. **Caching**: Store computed results
4. **Monitoring**: Add metrics and logging
5. **Rate Limiting**: Protect against API abuse

---

## Conclusion

MOFMaster demonstrates a clean, extensible architecture for scientific workflows:

✅ **Separation of Concerns**: Planning vs Execution  
✅ **Type Safety**: TypedDict + Pydantic  
✅ **Testability**: Pure functions, no hidden state  
✅ **Extensibility**: Easy to add tools and agents  
✅ **Scientific Rigor**: Validation before execution  

The system is production-ready while remaining simple enough to understand and modify.

---

## Quick Reference

### Key Files
- `app/graph.py`: Workflow definition
- `app/state.py`: Shared state
- `app/agents/runner.py`: Tool execution
- `app/tools/`: All tools
- `README_KNOWLEDGE.md`: Agent capabilities

### Key Concepts
- **State**: Shared memory (`AgentState`)
- **Node**: Agent function that transforms state
- **Edge**: Transition between nodes (conditional or direct)
- **Tool**: Python function wrapped with `@tool`
- **Plan**: List of tool names to execute

### Quick Commands
```bash
# Start server
uv run python app/server.py

# Run tests
uv run pytest

# Run examples
uv run python example_usage.py

# Format code
uv run black .
uv run ruff check .
```

---

**Happy coding! 🚀**
