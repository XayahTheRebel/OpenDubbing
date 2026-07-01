"""Workspace manager for intermediate results."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Workspace:
    """Manages all intermediate artifacts for a dubbing task."""

    SUBDIRECTORIES = (
        "input",
        "cache",
        "timeline",
        "translation",
        "tts",
        "face",
        "output",
    )

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create workspace subdirectories if they do not exist."""
        self.root.mkdir(parents=True, exist_ok=True)
        for name in self.SUBDIRECTORIES:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def path_for(self, step: str, name: str) -> Path:
        """Return a canonical path for an intermediate artifact.

        Args:
            step: The pipeline step or subdirectory name.
            name: The artifact file name.

        Returns:
            Resolved path under the workspace root.
        """
        if step not in self.SUBDIRECTORIES:
            raise ValueError(f"Unknown workspace step: {step}")
        return self.root / step / name

    def exists(self, step: str, name: str) -> bool:
        """Check whether an artifact already exists."""
        return self.path_for(step, name).exists()

    @property
    def timeline_path(self) -> Path:
        """Default timeline file path."""
        return self.root / "timeline" / "timeline.jsonl"

    @property
    def cache_path(self) -> Path:
        """Default cache directory."""
        return self.root / "cache"

    def relative(self, path: Path | str) -> Path:
        """Return a path relative to the workspace root."""
        return Path(path).resolve().relative_to(self.root)

    def to_dict(self) -> dict[str, Any]:
        """Serialize workspace metadata."""
        return {"root": str(self.root)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Workspace:
        """Deserialize workspace metadata."""
        return cls(root=data["root"])
