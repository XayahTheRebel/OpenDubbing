"""Pipeline orchestrator for executing engines in order."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opendubbing.core.interfaces import Engine
from opendubbing.core.registry import ProviderRegistry
from opendubbing.core.timeline import Timeline
from opendubbing.core.workspace import Workspace
from opendubbing.engines.base import EngineRegistry
from opendubbing.pipeline.cache import PipelineCache
from opendubbing.pipeline.errors import PipelineError, TaskCancelledError
from opendubbing.pipeline.logger import get_logger

logger = get_logger("pipeline.orchestrator")

DEFAULT_STEPS = [
    "input",
    "asr",
    "timeline",
    "translation",
    "length_optimizer",
    "prosody",
    "tts",
    "audio_post",
    "face",
    "video_post",
    "output",
]


class PipelineOrchestrator:
    """Execute a chain of engines with caching and progress callbacks."""

    def __init__(
        self,
        workspace: Workspace,
        config: dict[str, Any],
        engine_registry: EngineRegistry | None = None,
        provider_registry: ProviderRegistry | None = None,
    ) -> None:
        self.workspace = workspace
        self.config = config
        self.engine_registry = engine_registry or EngineRegistry()
        self.provider_registry = provider_registry or ProviderRegistry()
        self.cache = PipelineCache(workspace)
        self._cancelled = False
        self._progress_callbacks: list[Callable[[str, dict[str, Any]], None]] = []

    def add_progress_callback(
        self, callback: Callable[[str, dict[str, Any]], None]
    ) -> None:
        """Register a callback receiving step name and progress payload."""
        self._progress_callbacks.append(callback)

    def cancel(self) -> None:
        """Request cancellation of the running pipeline."""
        self._cancelled = True

    def _emit(self, step: str, status: str, **kwargs: Any) -> None:
        payload = {"status": status, **kwargs}
        for callback in self._progress_callbacks:
            callback(step, payload)

    def _build_engine(self, step: str) -> Engine:
        engine_config = dict(self.config)
        engine_config["step"] = step
        engine_config["registry"] = self.provider_registry
        return self.engine_registry.build(step, engine_config)

    def run(
        self,
        steps: list[str] | None = None,
        timeline: Timeline | None = None,
        resume: bool = False,
    ) -> Timeline:
        """Run the configured engine pipeline.

        Args:
            steps: Ordered list of engine names. Defaults to DEFAULT_STEPS.
            timeline: Initial timeline. Defaults to empty Timeline.
            resume: If True, skip already completed steps.

        Returns:
            The final timeline.

        Raises:
            PipelineError: If any engine fails.
            TaskCancelledError: If cancellation is requested.
        """
        steps = steps or DEFAULT_STEPS
        timeline = timeline or Timeline()

        if resume:
            logger.info("resuming_pipeline", workspace=str(self.workspace.root))

        for step in steps:
            if self._cancelled:
                raise TaskCancelledError(f"Pipeline cancelled at step {step}")

            if self.cache.should_skip(step, resume):
                logger.info("skipping_cached_step", step=step)
                self._emit(step, "skipped")
                continue

            if self.cache.hit(step, timeline):
                logger.info("cache_hit", step=step)
                timeline = self.cache.load(step)
                self._emit(step, "cached")
                continue

            engine = self._build_engine(step)
            logger.info("running_step", step=step)
            self._emit(step, "started")
            try:
                timeline = engine.process(timeline, self.workspace)
                engine.save(timeline, self.workspace)
                self.cache.commit(step, timeline)
                self._emit(step, "completed")
            except Exception as exc:
                self.cache.mark_failed(step)
                self._emit(step, "failed", error=str(exc))
                raise PipelineError(
                    f"Step {step} failed: {exc}", step=step
                ) from exc
            finally:
                engine.release()

        logger.info("pipeline_completed")
        return timeline

    def reset(self) -> None:
        """Reset cache state."""
        self.cache.reset()
