"""
LangSmith Configuration and Utilities

This module provides utilities for configuring and using LangSmith tracing.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()


def is_langsmith_enabled() -> bool:
    """
    Check if LangSmith tracing is enabled.

    Returns:
        True if LANGCHAIN_TRACING_V2 is set to 'true', False otherwise
    """
    return os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"


def get_langsmith_project() -> Optional[str]:
    """
    Get the LangSmith project name from environment variables.

    Returns:
        Project name if set, None otherwise
    """
    return os.getenv("LANGCHAIN_PROJECT")


def get_langsmith_api_key() -> Optional[str]:
    """
    Get the LangSmith API key from environment variables.

    Returns:
        API key if set, None otherwise
    """
    return os.getenv("LANGCHAIN_API_KEY")


def get_langsmith_config() -> Dict[str, Any]:
    """
    Get LangSmith configuration status.

    Returns:
        Dictionary with configuration status and values (without sensitive data)
    """
    return {
        "enabled": is_langsmith_enabled(),
        "project": get_langsmith_project(),
        "api_key_set": bool(get_langsmith_api_key()),
        "endpoint": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
    }


def create_run_config(
    thread_id: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a run configuration for LangGraph with LangSmith tracing metadata.

    Args:
        thread_id: Optional thread ID for conversation tracking
        run_name: Optional name for this run (appears in LangSmith UI)
        tags: Optional list of tags for filtering in LangSmith
        metadata: Optional metadata dictionary for additional context

    Returns:
        Configuration dictionary for LangGraph invoke/stream calls
    """
    config: Dict[str, Any] = {}

    if thread_id:
        config["configurable"] = {"thread_id": thread_id}

    if run_name:
        config["run_name"] = run_name

    if tags:
        config["tags"] = tags

    if metadata:
        config["metadata"] = metadata

    return config


def validate_langsmith_config() -> tuple[bool, list[str]]:
    """
    Validate LangSmith configuration and return status with any issues.

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    if not is_langsmith_enabled():
        return True, []  # Not enabled is fine, just return no issues

    # If tracing is enabled, check for required config
    if not get_langsmith_api_key():
        issues.append("LANGCHAIN_API_KEY is not set (required when LANGCHAIN_TRACING_V2=true)")

    return len(issues) == 0, issues


def print_langsmith_status() -> None:
    """
    Print LangSmith configuration status to console.
    Useful for startup logging.
    """
    config = get_langsmith_config()

    if config["enabled"]:
        print("=" * 60)
        print("LangSmith Tracing: ENABLED")
        print(f"  Project: {config['project'] or '(default)'}")
        print(f"  API Key: {'✓ Set' if config['api_key_set'] else '✗ Not Set'}")
        print(f"  Endpoint: {config['endpoint']}")
        print("=" * 60)
    else:
        print("=" * 60)
        print("LangSmith Tracing: DISABLED")
        print("  To enable, set LANGCHAIN_TRACING_V2=true in your .env file")
        print("=" * 60)
