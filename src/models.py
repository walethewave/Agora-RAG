"""Pydantic models for the FastAPI application.

Only the request and response shapes used by the live endpoints are kept here.
"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request body for the chat question endpoints."""

    entity_id: str = Field(..., description="Pinecone namespace for the request")
    question: str = Field(..., min_length=1, description="User question to answer")
    session_id: Optional[str] = Field(
        default=None,
        description="Upstash session identifier used for short-term memory",
    )


class CreateSessionRequest(BaseModel):
    """Request body for creating a new chat session."""

    entity_id: str = Field(..., description="Pinecone namespace for the session")


class APIResponse(BaseModel):
    """Standard response envelope returned by the API."""

    responseCode: str = Field(..., description="'00' means success, '01' means failure")
    responseMessage: str = Field(..., description="Human-readable status message")
    data: Optional[Any] = Field(default=None, description="Optional payload for the caller")