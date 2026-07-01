"""Application-level service orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opendubbing.config import AppConfig, load_config
from opendubbing.core.registry import create_default_registry
from opendubbing.core.workspace import Workspace
from opendubbing.engines.base import create_default_engine_registry
from opendubbing.pipeline.orchestrator import PipelineOrchestrator


def _build_config_dict(config: AppConfig, input_path: Path | None = None) -> dict[str, Any]:
    data = config.to_dict()
    if input_path:
        data["input_path"] = str(input_path)
    return data


def process_video(
    input_path: Path,
    config_path: Path,
    resume: bool = False,
    workspace_root: str | None = None,
) -> None:
    """Process a video through the dubbing pipeline."""
    app_config = load_config(config_path)
    root = workspace_root or app_config.workspace.root
    workspace = Workspace(root)
    config_dict = _build_config_dict(app_config, input_path)

    engine_registry = create_default_engine_registry()
    provider_registry = create_default_registry()

    orchestrator = PipelineOrchestrator(
        workspace=workspace,
        config=config_dict,
        engine_registry=engine_registry,
        provider_registry=provider_registry,
    )
    orchestrator.run(steps=app_config.pipeline.steps, resume=resume)
    print(f"Processing complete. Output in: {workspace.root / 'output'}")


def run_api_server(
    config_path: Path,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Start the OpenDubbing API server."""
    from opendubbing.api.server import start_server

    app_config = load_config(config_path)
    start_server(
        config=app_config,
        host=host or app_config.api.get("host", "127.0.0.1"),
        port=port or app_config.api.get("port", 8000),
    )


def list_providers() -> None:
    """Print all registered providers."""
    registry = create_default_registry()
    for kind in registry.list_kinds():
        print(kind)
        for name in registry.list_names(kind):
            print(f"  - {name}")
