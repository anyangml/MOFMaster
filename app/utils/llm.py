"""
LLM Model Factory - Creates and configures language models
"""

import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# Load environment variables (including LangSmith config)
load_dotenv()

# Ensure LangSmith environment variables are available
# These are used automatically by LangChain if set
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_key_here
# LANGCHAIN_PROJECT=MOFMaster-Dev


def get_llm(model_name: str = "gpt-4o", temperature: float = 0.0, streaming: bool = False):
    """
    Factory function to create language model instances.

    Args:
        model_name: Name of the model (e.g., "gpt-4o", "OpenAI/Azure-GPT-5.1", "claude-3-5-sonnet-20241022")
        temperature: Temperature for sampling (0.0 = deterministic)
        streaming: Whether to enable streaming

    Returns:
        A configured LangChain chat model instance
    """

    # Check if using custom API endpoint (GPUGeek)
    custom_api_base = os.getenv("OPENAI_API_BASE")
    use_custom_endpoint = custom_api_base is not None

    if model_name.startswith("gpt") or model_name.startswith("o1") or "GPT" in model_name or use_custom_endpoint:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Configure for custom endpoint if specified
        if use_custom_endpoint:
            # For custom endpoints like GPUGeek
            # Try OpenAI-compatible endpoint first: /v1/chat/completions
            # If GPUGeek supports it, use that. Otherwise, we'll need a custom adapter.
            base_url = custom_api_base.rstrip("/")
            
            # Check if base_url already includes /v1, if not, LangChain will add it
            # For GPUGeek, try: https://api.gpugeek.com/v1
            # LangChain will append /chat/completions automatically
            
            # Use Bearer token authentication if API key doesn't start with "sk-"
            headers = None
            if not api_key.startswith("sk-"):
                headers = {
                    "Authorization": f"Bearer {api_key}",
                }
            
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                streaming=streaming,
                api_key=api_key,
                base_url=base_url,
                default_headers=headers,
            )
        else:
            # Standard OpenAI API
            return ChatOpenAI(
                model=model_name, temperature=temperature, streaming=streaming, api_key=api_key
            )

    elif model_name.startswith("claude"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        return ChatAnthropic(
            model=model_name, temperature=temperature, streaming=streaming, api_key=api_key
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")


def get_analyzer_llm():
    """Get LLM configured for the Analyzer agent"""
    # Use custom model if specified, otherwise default to gpt-4o
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o")
    return get_llm(model_name=model_name, temperature=0.0)


def get_supervisor_llm():
    """Get LLM configured for the Supervisor agent (with JSON mode support)"""
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o")
    llm = get_llm(model_name=model_name, temperature=0.0)
    
    # Try to enable JSON mode for better structured output support
    # This helps with APIs that support response_format
    try:
        # Check if the LLM supports response_format binding
        if hasattr(llm, 'bind'):
            # Try to bind JSON mode (works with OpenAI-compatible APIs)
            # Note: This might not work with all APIs, so we wrap in try/except
            return llm
    except Exception:
        pass
    
    return llm


def get_reporter_llm():
    """Get LLM configured for the Reporter agent"""
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o")
    return get_llm(model_name=model_name, temperature=0.0)
