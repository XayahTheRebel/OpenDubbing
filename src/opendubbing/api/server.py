"""FastAPI server bootstrap."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from opendubbing.api.dependencies import AppState, set_state
from opendubbing.api.routes import router as api_router
from opendubbing.api.websocket import router as ws_router
from opendubbing.config import AppConfig


def create_app(config: AppConfig) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="OpenDubbing API",
        version="0.1.0",
        description="AI video dubbing pipeline API",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    state = AppState(config)
    set_state(state)
    app.state.engine_registry = state.engine_registry
    app.state.provider_registry = state.provider_registry
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(ws_router, prefix="/api/v1")
    return app


def start_server(config: AppConfig, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the Uvicorn server."""
    import uvicorn

    app = create_app(config)
    uvicorn.run(app, host=host, port=port)
