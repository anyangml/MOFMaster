"""
FastAPI Server with LangServe - API Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from app.graph import get_compiled_graph

# Create FastAPI app
app = FastAPI(
    title="MOF-Scientist Backend",
    version="1.0.0",
    description="Scientific workflow agent for Metal-Organic Framework computational chemistry",
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


# Add LangServe routes
add_routes(
    app,
    graph,
    path="/mof-scientist",
    enabled_endpoints=["invoke", "stream", "playground"],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
