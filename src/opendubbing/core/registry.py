"""Provider registry for discovering and building provider instances."""

from __future__ import annotations

from typing import Any, TypeVar

from opendubbing.core.interfaces import Provider, ProviderNotFoundError

T = TypeVar("T", bound=Provider)


class ProviderRegistry:
    """Registry mapping provider kinds and names to provider classes."""

    def __init__(self) -> None:
        self._providers: dict[str, dict[str, type[Provider]]] = {}

    def register(self, kind: str, name: str, provider_class: type[T]) -> type[T]:
        """Register a provider class.

        Args:
            kind: The capability category, e.g. "tts".
            name: The provider identifier, e.g. "cosyvoice2".
            provider_class: The provider class to register.

        Returns:
            The registered provider class (for decorator use).
        """
        self._providers.setdefault(kind, {})[name] = provider_class
        return provider_class

    def get(self, kind: str, name: str) -> type[Provider]:
        """Retrieve a provider class by kind and name."""
        try:
            return self._providers[kind][name]
        except KeyError as exc:
            raise ProviderNotFoundError(
                f"Provider not found: kind={kind}, name={name}"
            ) from exc

    def list_kinds(self) -> list[str]:
        """Return all registered provider kinds."""
        return sorted(self._providers.keys())

    def list_names(self, kind: str) -> list[str]:
        """Return all registered provider names for a kind."""
        return sorted(self._providers.get(kind, {}).keys())

    def build(self, kind: str, name: str, config: dict[str, Any] | None = None) -> Provider:
        """Instantiate and initialize a provider.

        Args:
            kind: Provider kind.
            name: Provider name.
            config: Provider-specific configuration.

        Returns:
            An initialized provider instance.
        """
        provider_class = self.get(kind, name)
        provider = provider_class()
        provider.initialize(config or {})
        return provider

    def decorator(self, kind: str, name: str) -> Any:
        """Decorator for registering provider classes.

        Usage:
            registry = ProviderRegistry()

            @registry.decorator("tts", "cosyvoice2")
            class CosyVoice2Provider(Provider): ...
        """

        def wrapper(provider_class: type[T]) -> type[T]:
            return self.register(kind, name, provider_class)

        return wrapper


def create_default_registry() -> ProviderRegistry:
    """Create a registry with all built-in providers pre-registered."""
    from opendubbing.providers.asr.mock_asr import MockASRProvider
    from opendubbing.providers.asr.qwen3_asr import Qwen3ASRProvider
    from opendubbing.providers.asr.whisper_asr import WhisperASRProvider
    from opendubbing.providers.audio_separation.demucs import DemucsProvider
    from opendubbing.providers.face.hallo3 import Hallo3Provider
    from opendubbing.providers.face.mock_face import MockFaceProvider
    from opendubbing.providers.forced_alignment.mock_forced_aligner import (
        MockForcedAlignerProvider,
    )
    from opendubbing.providers.forced_alignment.qwen3_forced_aligner import (
        Qwen3ForcedAlignerProvider,
    )
    from opendubbing.providers.noise_reduction.deep_filter_net3 import (
        DeepFilterNet3Provider,
    )
    from opendubbing.providers.translation.claude import ClaudeProvider
    from opendubbing.providers.translation.gemini import GeminiProvider
    from opendubbing.providers.translation.gpt import GPTProvider
    from opendubbing.providers.translation.mock_translation import (
        MockTranslationProvider,
    )
    from opendubbing.providers.translation.nllb import NLLBProvider
    from opendubbing.providers.tts.cosyvoice2 import CosyVoice2Provider
    from opendubbing.providers.tts.edge_tts import EdgeTTSProvider
    from opendubbing.providers.tts.mock_tts import MockTTSProvider
    from opendubbing.providers.vad.silero_vad import SileroVADProvider

    registry = ProviderRegistry()
    registry.register("audio_separation", "demucs", DemucsProvider)
    registry.register("noise_reduction", "deep_filter_net3", DeepFilterNet3Provider)
    registry.register("vad", "silero_vad", SileroVADProvider)
    registry.register("asr", "qwen3_asr", Qwen3ASRProvider)
    registry.register("asr", "whisper", WhisperASRProvider)
    registry.register("asr", "mock_asr", MockASRProvider)
    registry.register("forced_alignment", "qwen3_forced_aligner", Qwen3ForcedAlignerProvider)
    registry.register("forced_alignment", "mock_forced_aligner", MockForcedAlignerProvider)
    registry.register("translation", "nllb", NLLBProvider)
    registry.register("translation", "gpt", GPTProvider)
    registry.register("translation", "claude", ClaudeProvider)
    registry.register("translation", "gemini", GeminiProvider)
    registry.register("translation", "mock_translation", MockTranslationProvider)
    registry.register("tts", "cosyvoice2", CosyVoice2Provider)
    registry.register("tts", "edge_tts", EdgeTTSProvider)
    registry.register("tts", "mock_tts", MockTTSProvider)
    registry.register("face", "hallo3", Hallo3Provider)
    registry.register("face", "mock_face", MockFaceProvider)
    return registry
