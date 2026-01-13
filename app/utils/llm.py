"""
LLM Model Factory - Creates and configures language models
"""

import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_llm(model_name: str = "gpt-4o", temperature: float = 0.0, streaming: bool = False):
    """
    Factory function to create language model instances.

    Args:
        model_name: Name of the model (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        temperature: Temperature for sampling (0.0 = deterministic)
        streaming: Whether to enable streaming

    Returns:
        A configured LangChain chat model instance
    """

    if model_name.startswith("gpt") or model_name.startswith("o1"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

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
    return get_llm(model_name="gpt-4o", temperature=0.0)


def get_supervisor_llm():
    """Get LLM configured for the Supervisor agent"""
    return get_llm(model_name="gpt-4o", temperature=0.0)


def get_reporter_llm():
    """Get LLM configured for the Reporter agent"""
    return get_llm(model_name="gpt-4o", temperature=0.0)
