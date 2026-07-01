"""Face animation engine: generate lip-synced face video."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opendubbing.core.interfaces import Engine
from opendubbing.core.registry import ProviderRegistry
from opendubbing.core.timeline import Timeline
from opendubbing.core.workspace import Workspace


class FaceAnimationEngine(Engine):
    """Generate lip-sync video using a face animation provider."""

    name = "face"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config
        self.provider_config = config.get("providers", {}).get("face", {})
        self.provider_name = self.provider_config.get("name")
        self.registry = config.get("registry", ProviderRegistry())

    def process(self, timeline: Timeline, workspace: Workspace) -> Timeline:
        video = timeline.metadata.get("input_video")
        if not video:
            return timeline

        dub = timeline.metadata.get("dub_audio")
        if not dub:
            raise ValueError("dub_audio not available for face animation")

        if not self.provider_name:
            return timeline

        video_abs = workspace.root / video if not Path(video).is_absolute() else Path(video)
        dub_abs = workspace.root / dub if not Path(dub).is_absolute() else Path(dub)
        out = workspace.path_for("face", "face_video.mp4")

        provider = self.registry.build(
            "face", self.provider_name, self.provider_config
        )
        provider.load_model()
        try:
            provider.infer(
                {
                    "video": str(video_abs),
                    "audio": str(dub_abs),
                    "out_path": str(out),
                }
            )
        finally:
            provider.release()

        timeline.metadata["face_video"] = str(workspace.relative(out))
        return timeline

    def save(self, timeline: Timeline, workspace: Workspace) -> None:
        timeline.save(workspace.path_for("face", "face.jsonl"))

    def release(self) -> None:
        pass
