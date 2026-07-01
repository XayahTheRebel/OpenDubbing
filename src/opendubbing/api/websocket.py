"""WebSocket endpoint for pipeline progress."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from opendubbing.api.dependencies import get_tasks

router = APIRouter()


async def broadcast_progress(task_id: str, step: str, payload: dict[str, Any]) -> None:
    """Broadcast a progress event to all connected WebSocket clients."""
    tasks = get_tasks()
    task = tasks.get(task_id, {})
    connections = task.get("websocket_connections", [])
    message = {
        "task_id": task_id,
        "step": step,
        **payload,
    }
    dead = []
    for conn in connections:
        try:
            await conn.send_json(message)
        except Exception:  # pragma: no cover - websocket errors
            dead.append(conn)
    for conn in dead:
        connections.remove(conn)


@router.websocket("/ws/tasks/{task_id}")
async def task_websocket(websocket: WebSocket, task_id: str) -> None:
    """WebSocket endpoint for monitoring a task."""
    await websocket.accept()
    tasks = get_tasks()
    task = tasks.setdefault(
        task_id, {"websocket_connections": [], "status": "pending"}
    )
    task["websocket_connections"].append(websocket)
    try:
        await websocket.send_json({"task_id": task_id, "status": task["status"]})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        task["websocket_connections"].remove(websocket)
