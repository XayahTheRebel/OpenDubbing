"""TTS engine: synthesize dubbed audio segments."""

from __future__ import annotations

from typing import Any

from opendubbing.core.interfaces import Engine
from opendubbing.core.registry import ProviderRegistry
from opendubbing.core.timeline import Timeline
from opendubbing.core.workspace import Workspace


class TTSEngine(Engine):
    """Synthesize speech from translated text using a TTS provider."""

    name = "tts"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.provider_config = config.get("providers", {}).get("tts", {})
        self.provider_name = self.provider_config.get("name")
        self.registry = config.get("registry", ProviderRegistry())

    def process(self, timeline: Timeline, workspace: Workspace) -> Timeline:
        if not self.provider_name:
            raise ValueError("TTS provider name is required")

        provider = self.registry.build(
            "tts", self.provider_name, self.provider_config
        )
        provider.load_model()
        try:
            for sentence in timeline.sentences:
                if not sentence.translation:
                    continue
                out = workspace.path_for("tts", f"{sentence.id}.wav")
                result = provider.infer(
                    {
                        "text": sentence.translation,
                        "language": timeline.target_language,
                        "speech_rate": sentence.speech_rate,
                        "emotion": sentence.emotion,
                        "out_path": str(out),
                    }
                )
                sentence.metadata["tts_audio"] = str(workspace.relative(out))
                sentence.metadata["tts_duration"] = result.get("duration", 0.0)
        finally:
            provider.release()
        return timeline

    def save(self, timeline: Timeline, workspace: Workspace) -> None:
        timeline.save(workspace.path_for("timeline", "tts.jsonl"))

    def release(self) -> None:
        pass
