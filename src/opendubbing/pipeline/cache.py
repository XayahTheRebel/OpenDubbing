"""Pipeline cache for resumable execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from opendubbing.core.timeline import Timeline
from opendubbing.core.workspace import Workspace
from opendubbing.pipeline.errors import CacheError


class PipelineCache:
    """File-based cache enabling breakpoint-resume execution."""

    CACHE_VERSION = "1"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.cache_dir = workspace.cache_path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.cache_dir / "state.json"

    def _step_path(self, step: str) -> Path:
        return self.cache_dir / f"{step}.jsonl"

    def _state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"version": self.CACHE_VERSION, "completed": [], "failed": None}
        with self.state_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_state(self, state: dict[str, Any]) -> None:
        with self.state_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def _hash(self, timeline: Timeline) -> str:
        data = json.dumps(timeline.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def hit(self, step: str, timeline: Timeline) -> bool:
        """Check whether a cached result exists and is valid for the step."""
        state = self._state()
        if step not in state.get("completed", []):
            return False
        step_path = self._step_path(step)
        if not step_path.exists():
            return False
        metadata_path = self.cache_dir / f"{step}.meta.json"
        if not metadata_path.exists():
            return False
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata.get("timeline_hash") == self._hash(timeline)

    def load(self, step: str) -> Timeline:
        """Load a cached timeline for a step."""
        step_path = self._step_path(step)
        try:
            return Timeline.load(step_path)
        except Exception as exc:
            raise CacheError(f"Failed to load cache for {step}") from exc

    def commit(self, step: str, timeline: Timeline) -> None:
        """Persist a timeline as the cached result for a step."""
        step_path = self._step_path(step)
        try:
            timeline.save(step_path)
        except Exception as exc:
            raise CacheError(f"Failed to save cache for {step}") from exc

        metadata = {
            "version": self.CACHE_VERSION,
            "timeline_hash": self._hash(timeline),
        }
        metadata_path = self.cache_dir / f"{step}.meta.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        state = self._state()
        if step not in state["completed"]:
            state["completed"].append(step)
        state["failed"] = None
        self._save_state(state)

    def mark_failed(self, step: str) -> None:
        """Mark a step as failed in cache state."""
        state = self._state()
        state["failed"] = step
        self._save_state(state)

    def last_completed(self) -> str | None:
        """Return the last successfully completed step, if any."""
        state = self._state()
        completed = state.get("completed", [])
        return completed[-1] if completed else None

    def reset(self) -> None:
        """Clear all cached state and artifacts."""
        for path in self.cache_dir.glob("*"):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                Path(path).rmdir()
        self._save_state({"version": self.CACHE_VERSION, "completed": [], "failed": None})

    def should_skip(self, step: str, resume: bool) -> bool:
        """Determine whether a step should be skipped during resume."""
        if not resume:
            return False
        last = self.last_completed()
        if last is None:
            return False
        state = self._state()
        completed = state.get("completed", [])
        failed = state.get("failed")
        if failed and step != failed:
            return step in completed
        return step in completed
