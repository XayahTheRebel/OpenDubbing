"""Output engine: finalize and deliver output artifacts."""

from __future__ import annotations

import shutil
from typing import Any

from opendubbing.core.interfaces import Engine
from opendubbing.core.timeline import Timeline
from opendubbing.core.workspace import Workspace


class OutputEngine(Engine):
    """Copy final artifacts to the workspace output directory."""

    name = "output"

    def initialize(self, config: dict[str, Any]) -> None:
        self.config = config

    def process(self, timeline: Timeline, workspace: Workspace) -> Timeline:
        final = workspace.path_for("output", "final.mp4")
        filename = self.config.get("output", {}).get("filename", "output.mp4")
        dst = workspace.path_for("output", filename)

        if final != dst:
            if dst.exists():
                dst.unlink()
            shutil.copyfile(final, dst)

        timeline.metadata["output"] = str(workspace.relative(dst))
        return timeline

    def save(self, timeline: Timeline, workspace: Workspace) -> None:
        timeline.save(workspace.path_for("output", "timeline.jsonl"))

    def release(self) -> None:
        pass
