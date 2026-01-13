"""
FastAPI Server with LangServe - API Entry Point
"""

# IMPORTANT: Load environment variables FIRST, before importing anything else
# This ensures LangSmith tracing is configured before LangChain components are initialized
from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langserve import add_routes
from langchain_core.messages import convert_to_messages
import traceback

from app.graph import get_compiled_graph
from app.utils.langsmith_config import (
    validate_langsmith_config,
    print_langsmith_status,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    is_valid, issues = validate_langsmith_config()
    if not is_valid:
        print("⚠️  LangSmith Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
        print("   Traces may not be sent to LangSmith.")
    print_langsmith_status()
    yield
    # Shutdown (if needed)


# Check for debug mode
import os
debug_mode = os.getenv("DEBUG", "false").lower() == "true"

# Create FastAPI app with lifespan
app = FastAPI(
    title="MOF-Scientist Backend",
    version="1.0.0",
    description="Scientific workflow agent for Metal-Organic Framework computational chemistry",
    lifespan=lifespan,
    debug=debug_mode,  # Enable FastAPI debug mode
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the compiled graph
graph = get_compiled_graph()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to log errors and return detailed error messages."""
    error_traceback = traceback.format_exc()
    print(f"\n❌ Error occurred: {type(exc).__name__}: {str(exc)}")
    print(f"📋 Traceback:\n{error_traceback}")
    # Show full traceback in debug mode
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "detail": error_traceback if debug_mode else "Internal server error. Check server logs for details. Set DEBUG=true to see full traceback.",
        },
    )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MOF-Scientist Backend API",
        "version": "1.0.0",
        "endpoints": {
            "invoke": "/mof-scientist/invoke",
            "stream": "/mof-scientist/stream",
            "playground": "/mof-scientist/playground",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


# Create a wrapper function to handle message format conversion
def convert_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert API input format (role-based) to LangChain message format (type-based)
    and initialize all required state fields with default values."""
    result = {
        # Initialize all required state fields with defaults
        "original_query": "",
        "plan": [],
        "current_step": 0,
        "tool_outputs": {},
        "review_feedback": "",
        "is_plan_approved": False,
        "_rejection_count": 0,  # Track rejections to prevent infinite loops
    }
    
    if "messages" in input_data:
        # Convert role-based messages to LangChain message objects
        # First convert role to type format, then use convert_to_messages
        messages_to_convert = []
        for msg in input_data["messages"]:
            if isinstance(msg, dict):
                # Map role to type for LangChain compatibility
                role = msg.get("role", "user")
                content = msg.get("content", "")
                # Convert role to LangChain message type
                if role in ["user", "human"]:
                    messages_to_convert.append({"type": "human", "content": content})
                elif role in ["assistant", "ai"]:
                    messages_to_convert.append({"type": "ai", "content": content})
                elif role == "system":
                    messages_to_convert.append({"type": "system", "content": content})
                else:
                    messages_to_convert.append({"type": "human", "content": content})
            else:
                # Already in correct format (might be a message object or dict)
                messages_to_convert.append(msg)
        # Convert to proper LangChain message objects
        result["messages"] = convert_to_messages(messages_to_convert)
    else:
        # If no messages, use empty list
        result["messages"] = []
    
    # Merge any other fields from input_data
    for key, value in input_data.items():
        if key not in result:
            result[key] = value
    
    return result


# Wrap the graph to handle input conversion using RunnableLambda
from langchain_core.runnables import RunnableLambda

def convert_input_runnable(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Runnable that converts role-based messages to type-based format"""
    return convert_input(input_data)

# Create a runnable chain: convert input -> graph
input_converter = RunnableLambda(convert_input_runnable)
wrapped_graph = input_converter | graph

# Add LangServe routes
add_routes(
    app,
    wrapped_graph,
    path="/mof-scientist",
    enabled_endpoints=["invoke", "stream", "playground"],
)


if __name__ == "__main__":
    import uvicorn
    import os

    # Check for debug mode from environment variable
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    
    # Run server with debug options if enabled
    # When using reload=True, uvicorn requires the app as an import string
    if debug_mode:
        uvicorn.run(
            "app.server:app",  # Import string format: module:variable
            host="0.0.0.0",
            port=8000,
            reload=True,  # Auto-reload on code changes
            log_level="debug",  # More verbose logging
        )
    else:
        uvicorn.run(
            app,  # Can use app object directly when reload=False
            host="0.0.0.0",
            port=8000,
            log_level="info",
        )
