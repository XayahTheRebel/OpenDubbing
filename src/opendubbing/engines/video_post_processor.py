"""Video post processor: combine video and dubbed audio."""

from __future__ import annotations

from typing import Any

from opendubbing.core.interfaces import Engine
from opendubbing.core.timeline import Timeline
from opendubbing.core.workspace import Workspace
from opendubbing.utils import media


class VideoPostProcessor(Engine):
    """Merge face/video stream with dubbed audio into the final video."""

    name = "video_post"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config

    def process(self, timeline: Timeline, workspace: Workspace) -> Timeline:
        face_video = timeline.metadata.get("face_video")
        input_video = timeline.metadata.get("input_video")
        dub_audio = timeline.metadata.get("dub_audio")

        if not dub_audio:
            raise ValueError("dub_audio not available for video post processing")

        if face_video:
            video = workspace.root / face_video
        elif input_video:
            video = workspace.root / input_video
        else:
            raise ValueError("No video source available")

        dub = workspace.root / dub_audio
        out = workspace.path_for("output", "final.mp4")
        codec = self.config.get("output", {}).get("codec", "libx264")

        media.mux_audio_video(video, dub, out, codec=codec)
        timeline.metadata["final_video"] = str(workspace.relative(out))
        return timeline

    def save(self, timeline: Timeline, workspace: Workspace) -> None:
        timeline.save(workspace.path_for("timeline", "video_post.jsonl"))

    def release(self) -> None:
        pass
