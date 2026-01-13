# MOFMaster

MOFMaster is a modular multi-agent framework for metal–organic framework (MOF) research, integrating literature understanding, computational modeling, and extensible experiment interfaces. It is designed to orchestrate reading agents, calculation agents, and future experiment agents within a unified, extensible architecture.

## MOF-Scientist Backend

The MOF-Scientist Backend is a scientific workflow agent designed to democratize computational chemistry for Metal-Organic Frameworks (MOFs). The system accepts natural language queries, validates them against scientific constraints, and executes multi-step simulation workflows using a deterministic agentic loop.

### Architecture

The system is built using:
- **LangGraph**: State graph orchestration
- **LangServe**: REST API exposure via FastAPI
- **ASE** (Atomic Simulation Environment): Atomistic simulations
- **Pymatgen**: Materials science utilities

### Features

- Natural language query processing
- Automated workflow planning and validation
- MOF structure search
- Geometry optimization
- Energy and force calculations
- Structured result reporting

### Quick Start

1. **Install dependencies**:
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

3. **Run the server**:
```bash
uv run python -m app.server
# Or
python app/server.py
```

4. **Access the API**:
- API Documentation: http://localhost:8000/docs
- Playground: http://localhost:8000/mof-scientist/playground
- Health Check: http://localhost:8000/health

### API Endpoints

- `POST /mof-scientist/invoke` - Run the full workflow (blocking)
- `POST /mof-scientist/stream` - Stream workflow events
- `GET /mof-scientist/playground` - Interactive testing interface

### Example Usage

```bash
curl -X POST "http://localhost:8000/mof-scientist/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [
        {"role": "user", "content": "Find a copper-based MOF and calculate its energy"}
      ]
    }
  }'
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=app

# Run specific test file
uv run pytest tests/unit/test_tools.py
```

### Project Structure

```
mof-backend/
├── app/
│   ├── agents/          # Agent nodes (Analyzer, Supervisor, Runner, Reporter)
│   ├── tools/           # Scientific tools (search, optimize, calculate)
│   ├── utils/           # Utilities (LLM factory)
│   ├── graph.py         # Main LangGraph definition
│   ├── server.py        # FastAPI + LangServe entry point
│   ├── state.py         # AgentState TypedDict
│   └── schema.py        # Pydantic models
├── tests/               # Test suite
├── data/                # Local CIF files and results
├── README_KNOWLEDGE.md  # Agent knowledge base
└── pyproject.toml       # Dependencies
```

### Development

- **Linting**: `uv run ruff check app/`
- **Format**: `uv run black app/`
- **Type checking**: Install mypy with `uv pip install mypy`, then run `uv run mypy app/`

### License

See LICENSE file for details.
