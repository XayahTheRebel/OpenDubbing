"""Core abstractions for engines and providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from opendubbing.core.timeline import Timeline
from opendubbing.core.workspace import Workspace


class Engine(ABC):
    """Base class for all pipeline engines.

    Engines operate on a shared Timeline and Workspace. They must not call
    other engines or providers directly; orchestration is performed by the
    Pipeline.
    """

    name: str = ""

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the engine with task configuration."""

    @abstractmethod
    def process(self, timeline: Timeline, workspace: Workspace) -> Timeline:
        """Process the timeline and return an updated timeline."""

    @abstractmethod
    def save(self, timeline: Timeline, workspace: Workspace) -> None:
        """Persist intermediate results to the workspace."""

    @abstractmethod
    def release(self) -> None:
        """Release resources acquired by the engine."""


class Provider(ABC):
    """Base class for all model providers.

    A provider wraps a concrete model or external service and exposes a uniform
    inference interface. Providers must not call other providers.
    """

    name: str = ""
    kind: str = ""

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize provider configuration."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the underlying model or service resources."""

    @abstractmethod
    def infer(self, inputs: Any) -> Any:
        """Run inference on the given inputs and return outputs."""

    @abstractmethod
    def release(self) -> None:
        """Release model resources."""


class ProviderModelLoadError(Exception):
    """Raised when a provider fails to load its model."""


class ProviderNotFoundError(Exception):
    """Raised when a requested provider is not registered."""


class EngineNotFoundError(Exception):
    """Raised when a requested engine is not registered."""
