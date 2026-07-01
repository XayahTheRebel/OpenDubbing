"""Audio post processor: mix and enhance dubbed audio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from opendubbing.core.interfaces import Engine
from opendubbing.core.timeline import Timeline
from opendubbing.core.workspace import Workspace
from opendubbing.utils import media


class AudioPostProcessor(Engine):
    """Combine TTS segments, reduce noise, and normalize audio."""

    name = "audio_post"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config

    def process(self, timeline: Timeline, workspace: Workspace) -> Timeline:
        duration = timeline.metadata.get("duration", 0.0)
        sample_rate = timeline.metadata.get("sample_rate", 16000)
        if duration <= 0:
            raise ValueError("Input duration not available in timeline metadata")

        total_samples = int(duration * sample_rate)
        mix = np.zeros(total_samples, dtype=np.float32)

        for sentence in timeline.sentences:
            audio_rel = sentence.metadata.get("tts_audio")
            if not audio_rel:
                continue
            audio_path = (
                workspace.root / audio_rel
                if not Path(audio_rel).is_absolute()
                else Path(audio_rel)
            )
            samples, sr = media.read_audio(audio_path, sample_rate=sample_rate)
            start_sample = int(sentence.start * sample_rate)
            end_sample = min(start_sample + len(samples), total_samples)
            write_len = end_sample - start_sample
            if write_len > 0:
                mix[start_sample:end_sample] += samples[:write_len]

        peak = np.max(np.abs(mix))
        if peak > 0:
            mix = mix / peak * 0.95

        out = workspace.path_for("tts", "dub_full.wav")
        media.write_audio(mix, sample_rate, out)
        timeline.metadata["dub_audio"] = str(workspace.relative(out))
        return timeline

    def save(self, timeline: Timeline, workspace: Workspace) -> None:
        timeline.save(workspace.path_for("timeline", "audio_post.jsonl"))

    def release(self) -> None:
        pass
