"""Configuration models and loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for a single provider."""

    name: str
    model: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Pipeline execution configuration."""

    steps: list[str] = Field(default_factory=lambda: [
        "input",
        "asr",
        "timeline",
        "translation",
        "length_optimizer",
        "prosody",
        "tts",
        "audio_post",
        "face",
        "video_post",
        "output",
    ])


class WorkspaceConfig(BaseModel):
    """Workspace configuration."""

    root: str = "./workspace"


class OutputConfig(BaseModel):
    """Output file configuration."""

    filename: str = "output.mp4"
    format: str = "mp4"
    codec: str = "libx264"


class AppConfig(BaseModel):
    """Top-level application configuration."""

    target_language: str = "zh"
    sample_rate: int = 16000
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    output: OutputConfig = Field(default_factory=OutputConfig)
    api: dict[str, Any] = Field(default_factory=lambda: {"host": "127.0.0.1", "port": 8000})

    @classmethod
    def from_file(cls, path: Path | str) -> AppConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a dictionary."""
        return self.model_dump()


def load_config(path: Path | str) -> AppConfig:
    """Convenience function to load configuration."""
    return AppConfig.from_file(path)
