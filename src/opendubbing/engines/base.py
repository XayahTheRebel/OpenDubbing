"""Base engine utilities and registry."""

from __future__ import annotations

from typing import Any

from opendubbing.core.interfaces import Engine, EngineNotFoundError


class EngineRegistry:
    """Registry for engine classes."""

    def __init__(self) -> None:
        self._engines: dict[str, type[Engine]] = {}

    def register(self, name: str, engine_class: type[Engine]) -> type[Engine]:
        """Register an engine class by name."""
        self._engines[name] = engine_class
        return engine_class

    def get(self, name: str) -> type[Engine]:
        """Retrieve an engine class by name."""
        try:
            return self._engines[name]
        except KeyError as exc:
            raise EngineNotFoundError(f"Engine not found: {name}") from exc

    def list_names(self) -> list[str]:
        """Return all registered engine names."""
        return sorted(self._engines.keys())

    def build(self, name: str, config: dict[str, Any] | None = None) -> Engine:
        """Instantiate and initialize an engine."""
        engine_class = self.get(name)
        engine = engine_class()
        engine.initialize(config or {})
        return engine


def create_default_engine_registry() -> EngineRegistry:
    """Create a registry with all built-in engines pre-registered."""
    from opendubbing.engines.asr_engine import ASREngine
    from opendubbing.engines.audio_post_processor import AudioPostProcessor
    from opendubbing.engines.face_animation_engine import FaceAnimationEngine
    from opendubbing.engines.input_engine import InputEngine
    from opendubbing.engines.length_optimizer import LengthOptimizer
    from opendubbing.engines.output_engine import OutputEngine
    from opendubbing.engines.prosody_engine import ProsodyEngine
    from opendubbing.engines.timeline_engine import TimelineEngine
    from opendubbing.engines.translation_engine import TranslationEngine
    from opendubbing.engines.tts_engine import TTSEngine
    from opendubbing.engines.video_post_processor import VideoPostProcessor

    registry = EngineRegistry()
    registry.register("input", InputEngine)
    registry.register("asr", ASREngine)
    registry.register("timeline", TimelineEngine)
    registry.register("translation", TranslationEngine)
    registry.register("length_optimizer", LengthOptimizer)
    registry.register("prosody", ProsodyEngine)
    registry.register("tts", TTSEngine)
    registry.register("audio_post", AudioPostProcessor)
    registry.register("face", FaceAnimationEngine)
    registry.register("video_post", VideoPostProcessor)
    registry.register("output", OutputEngine)
    return registry
