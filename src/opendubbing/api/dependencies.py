"""Dependency injection for the API layer."""

from __future__ import annotations

from typing import Any

from opendubbing.config import AppConfig
from opendubbing.core.registry import ProviderRegistry, create_default_registry
from opendubbing.engines.base import EngineRegistry, create_default_engine_registry


class AppState:
    """Shared application state for the API server."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.engine_registry = create_default_engine_registry()
        self.provider_registry = create_default_registry()
        self.tasks: dict[str, dict[str, Any]] = {}


# Singleton state instance set during server startup
_state: AppState | None = None


def set_state(state: AppState) -> None:
    """Set the global application state."""
    global _state
    _state = state


def get_state() -> AppState:
    """Get the global application state."""
    if _state is None:
        raise RuntimeError("Application state has not been initialized")
    return _state


def get_engine_registry() -> EngineRegistry:
    """Get the engine registry from app state."""
    return get_state().engine_registry


def get_provider_registry() -> ProviderRegistry:
    """Get the provider registry from app state."""
    return get_state().provider_registry


def get_tasks() -> dict[str, dict[str, Any]]:
    """Get the task store from app state."""
    return get_state().tasks


def get_config() -> AppConfig:
    """Get the application configuration."""
    return get_state().config
