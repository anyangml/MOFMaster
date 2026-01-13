"""
Pydantic models for Input/Output schemas
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class UserMessage(BaseModel):
    """A message from the user"""

    role: str = Field(default="user", description="Message role")
    content: str = Field(..., description="Message content")


class ChatInput(BaseModel):
    """Input schema for the MOF-Scientist API"""

    messages: List[UserMessage] = Field(..., description="List of chat messages")


class ChatOutput(BaseModel):
    """Output schema for the MOF-Scientist API"""

    content: str = Field(..., description="The final report/response")
    plan: Optional[List[str]] = Field(None, description="The execution plan")
    tool_outputs: Optional[dict] = Field(None, description="Raw tool outputs")


class SupervisorReview(BaseModel):
    """Structured output from the Supervisor agent"""

    approved: bool = Field(..., description="Whether the plan is approved")
    feedback: str = Field(..., description="Explanation of the decision")
