"""Timeline engine: refine sentence/word timings using forced alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opendubbing.core.interfaces import Engine
from opendubbing.core.registry import ProviderRegistry
from opendubbing.core.timeline import Phoneme, Timeline, Word
from opendubbing.core.workspace import Workspace


class TimelineEngine(Engine):
    """Refine timeline timings with a forced alignment provider."""

    name = "timeline"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.provider_config = config.get("providers", {}).get(
            "forced_alignment", {}
        )
        self.provider_name = self.provider_config.get("name")
        self.registry = config.get("registry", ProviderRegistry())

    def process(self, timeline: Timeline, workspace: Workspace) -> Timeline:
        if not self.provider_name:
            return timeline

        audio_path = Path(timeline.metadata["input_audio"])
        audio_abs = (
            workspace.root / audio_path
            if not audio_path.is_absolute()
            else audio_path
        )

        provider = self.registry.build(
            "forced_alignment", self.provider_name, self.provider_config
        )
        provider.load_model()
        try:
            for sentence in timeline.sentences:
                if not sentence.words:
                    continue
                result = provider.infer(
                    {
                        "audio": str(audio_abs),
                        "text": sentence.text,
                        "words": [w.text for w in sentence.words],
                        "start": sentence.start,
                        "end": sentence.end,
                    }
                )
                aligned_words = []
                for w in result.get("words", []):
                    phonemes = [
                        Phoneme(
                            symbol=p["symbol"],
                            start=p["start"],
                            end=p["end"],
                            duration=p["end"] - p["start"],
                        )
                        for p in w.get("phonemes", [])
                    ]
                    aligned_words.append(
                        Word(
                            text=w["text"],
                            start=w["start"],
                            end=w["end"],
                            duration=w["end"] - w["start"],
                            phonemes=phonemes,
                        )
                    )
                if aligned_words:
                    sentence.words = aligned_words
                    sentence.start = aligned_words[0].start
                    sentence.end = aligned_words[-1].end
                    sentence.duration = sentence.end - sentence.start
                sentence.confidence = min(
                    sentence.confidence,
                    result.get("confidence", sentence.confidence),
                )
        finally:
            provider.release()
        return timeline

    def save(self, timeline: Timeline, workspace: Workspace) -> None:
        timeline.save(workspace.path_for("timeline", "aligned.jsonl"))

    def release(self) -> None:
        pass
