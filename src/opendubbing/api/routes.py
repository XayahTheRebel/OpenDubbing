"""REST API routes for OpenDubbing."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from opendubbing.api.dependencies import get_config, get_provider_registry, get_tasks
from opendubbing.api.schemas import (
    CreateTaskRequest,
    ProviderInfo,
    TaskResponse,
    TaskStatus,
)
from opendubbing.api.websocket import broadcast_progress
from opendubbing.config import AppConfig
from opendubbing.core.registry import ProviderRegistry
from opendubbing.core.workspace import Workspace
from opendubbing.engines.base import create_default_engine_registry
from opendubbing.pipeline.orchestrator import PipelineOrchestrator

router = APIRouter()


def _get_registry_from_request(request: Request) -> ProviderRegistry:
    return request.app.state.provider_registry


def _build_task_response(task_id: str, task: dict[str, Any]) -> TaskResponse:
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus(task.get("status", "pending")),
        current_step=task.get("current_step"),
        progress=task.get("progress", 0),
        error=task.get("error"),
        output_path=task.get("output_path"),
    )


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/providers")
async def list_providers(request: Request) -> list[ProviderInfo]:
    """List all registered providers."""
    registry = _get_registry_from_request(request)
    providers = []
    for kind in registry.list_kinds():
        for name in registry.list_names(kind):
            providers.append(ProviderInfo(kind=kind, name=name))
    return providers


@router.post("/tasks", response_model=TaskResponse)
async def create_task(
    request: CreateTaskRequest,
    app_config: AppConfig = Depends(get_config),
    provider_registry: ProviderRegistry = Depends(get_provider_registry),
) -> TaskResponse:
    """Create and start a new dubbing task."""
    task_id = str(uuid.uuid4())
    workspace = Workspace(Path(app_config.workspace.root) / task_id)
    tasks = get_tasks()
    tasks[task_id] = {
        "status": TaskStatus.PENDING.value,
        "workspace": str(workspace.root),
        "progress": 0,
        "current_step": None,
        "error": None,
        "output_path": None,
        "websocket_connections": [],
    }

    async def run() -> None:
        task_config = AppConfig.from_file(request.config_path)
        config_dict = task_config.to_dict()
        config_dict["input_path"] = request.input_path
        orchestrator = PipelineOrchestrator(
            workspace=workspace,
            config=config_dict,
            engine_registry=create_default_engine_registry(),
            provider_registry=provider_registry,
        )

        total = len(app_config.pipeline.steps)

        def on_progress(step: str, payload: dict[str, Any]) -> None:
            status = payload.get("status", "started")
            task = tasks[task_id]
            task["current_step"] = step
            if status == "completed":
                task["progress"] = min(100, int((app_config.pipeline.steps.index(step) + 1) / total * 100))
            asyncio.create_task(broadcast_progress(task_id, step, payload))

        orchestrator.add_progress_callback(on_progress)
        task = tasks[task_id]
        task["status"] = TaskStatus.RUNNING.value
        try:
            orchestrator.run(steps=task_config.pipeline.steps, resume=request.resume)
            task["status"] = TaskStatus.COMPLETED.value
            task["output_path"] = str(workspace.root / "output")
        except Exception as exc:  # pragma: no cover - broad error capture
            task["status"] = TaskStatus.FAILED.value
            task["error"] = str(exc)

    asyncio.create_task(run())
    return _build_task_response(task_id, tasks[task_id])


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str) -> TaskResponse:
    """Get the status of a task."""
    tasks = get_tasks()
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return _build_task_response(task_id, task)


@router.post("/tasks/{task_id}/cancel", response_model=TaskResponse)
async def cancel_task(task_id: str) -> TaskResponse:
    """Cancel a running task."""
    tasks = get_tasks()
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    # TODO: propagate cancellation to orchestrator
    task["status"] = TaskStatus.CANCELLED.value
    return _build_task_response(task_id, task)
