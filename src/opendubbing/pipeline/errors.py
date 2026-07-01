"""Pipeline exceptions."""

from __future__ import annotations


class PipelineError(Exception):
    """Base exception for pipeline execution errors."""

    def __init__(self, message: str, step: str | None = None) -> None:
        super().__init__(message)
        self.step = step


class CacheError(Exception):
    """Raised when cache read/write fails."""


class TaskCancelledError(Exception):
    """Raised when a running task is cancelled."""
