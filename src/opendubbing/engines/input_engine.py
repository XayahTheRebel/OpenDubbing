"""Input engine: load input media and extract source audio."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from opendubbing.core.interfaces import Engine
from opendubbing.core.timeline import Timeline
from opendubbing.core.workspace import Workspace
from opendubbing.utils import media


class InputEngine(Engine):
    """Prepare input video/audio for the pipeline."""

    name = "input"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.input_path = config.get("input_path")
        self.target_sample_rate = config.get("sample_rate", 16000)

    def process(self, timeline: Timeline, workspace: Workspace) -> Timeline:
        if not self.input_path:
            raise ValueError("input_path is required")
        input_path = Path(self.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        audio_out = workspace.path_for("input", "audio.wav")
        video_out = workspace.path_for("input", "video.mp4")

        if not audio_out.exists():
            media.extract_audio(
                input_path, audio_out, sample_rate=self.target_sample_rate
            )

        suffix = input_path.suffix.lower()
        is_video = suffix in {".mp4", ".mov", ".mkv", ".avi", ".webm"}
        if is_video and not video_out.exists():
            shutil.copyfile(input_path, video_out)

        timeline.metadata["input_audio"] = str(workspace.relative(audio_out))
        timeline.metadata["input_video"] = (
            str(workspace.relative(video_out)) if video_out.exists() else None
        )
        timeline.metadata["duration"] = media.probe_duration(audio_out)
        timeline.metadata["sample_rate"] = self.target_sample_rate
        return timeline

    def save(self, timeline: Timeline, workspace: Workspace) -> None:
        timeline.save(workspace.timeline_path)

    def release(self) -> None:
        pass
