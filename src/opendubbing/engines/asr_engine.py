"""ASR engine: transcribe source audio into a timeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opendubbing.core.interfaces import Engine
from opendubbing.core.registry import ProviderRegistry
from opendubbing.core.timeline import Sentence, Timeline, Word
from opendubbing.core.workspace import Workspace


class ASREngine(Engine):
    """Convert source audio to text timeline using an ASR provider."""

    name = "asr"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.provider_config = config.get("providers", {}).get("asr", {})
        self.provider_name = self.provider_config.get("name")
        self.registry = config.get("registry", ProviderRegistry())

    def process(self, timeline: Timeline, workspace: Workspace) -> Timeline:
        if not self.provider_name:
            raise ValueError("ASR provider name is required")

        audio_path = Path(timeline.metadata["input_audio"])
        audio_abs = (
            workspace.root / audio_path
            if not audio_path.is_absolute()
            else audio_path
        )

        provider = self.registry.build(
            "asr", self.provider_name, self.provider_config
        )
        provider.load_model()
        try:
            result = provider.infer(
                {
                    "audio": str(audio_abs),
                    "language": self.provider_config.get(
                        "options", {}
                    ).get("source_language", "auto"),
                }
            )
        finally:
            provider.release()

        timeline.source_language = result.get(
            "language", timeline.source_language
        )
        for i, seg in enumerate(result.get("segments", [])):
            words = [
                Word(
                    text=w["text"],
                    start=w["start"],
                    end=w["end"],
                    duration=w["end"] - w["start"],
                )
                for w in seg.get("words", [])
            ]
            sentence = Sentence(
                id=f"s{i:04d}",
                text=seg["text"].strip(),
                start=seg["start"],
                end=seg["end"],
                duration=seg["end"] - seg["start"],
                confidence=seg.get("confidence", 1.0),
                words=words,
            )
            timeline.append(sentence)
        return timeline

    def save(self, timeline: Timeline, workspace: Workspace) -> None:
        timeline.save(workspace.path_for("timeline", "asr.jsonl"))

    def release(self) -> None:
        pass
