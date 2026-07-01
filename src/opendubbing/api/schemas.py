"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    """Pipeline task status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CreateTaskRequest(BaseModel):
    """Request body for creating a dubbing task."""

    input_path: str
    config_path: str
    resume: bool = False


class TaskResponse(BaseModel):
    """Response model for a task."""

    task_id: str
    status: TaskStatus
    current_step: str | None = None
    progress: int = Field(0, ge=0, le=100)
    error: str | None = None
    output_path: str | None = None


class ProviderInfo(BaseModel):
    """Information about a registered provider."""

    kind: str
    name: str


class ProgressEvent(BaseModel):
    """Progress event pushed via WebSocket."""

    task_id: str
    step: str
    status: str
    progress: int = 0
    detail: dict[str, Any] = Field(default_factory=dict)
