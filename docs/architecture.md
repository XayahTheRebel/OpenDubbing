# OpenDubbing Architecture

## Overview

OpenDubbing follows a strict layered architecture:

```
Application → Pipeline → Engine → Provider → Model
```

This separation guarantees that future capabilities (live streaming, digital humans, plugin marketplace) can be added without refactoring existing code.

## Layers

### Application

- `cli.py`: Command-line interface.
- `application.py`: High-level service orchestration.
- `api/`: FastAPI server exposing HTTP and WebSocket endpoints.

The Application layer is the only layer allowed to interact with users or external clients.

### Pipeline

- `orchestrator.py`: Executes engines in order, handles caching, resumption, errors, and progress callbacks.
- `cache.py`: File-based cache supporting breakpoint-resume execution.
- `logger.py`: Structured logging.
- `errors.py`: Pipeline-specific exceptions.

The Pipeline is the sole scheduler. It never contains model inference code.

### Engine

Engines implement the `Engine` interface:

```python
class Engine(ABC):
    def initialize(self, config): ...
    def process(self, timeline, workspace): ...
    def save(self, timeline, workspace): ...
    def release(self): ...
```

Engines read and update the shared `Timeline` and persist intermediate results to the `Workspace`. They never call other engines or providers directly.

### Provider

Providers implement the `Provider` interface:

```python
class Provider(ABC):
    def initialize(self, config): ...
    def load_model(self): ...
    def infer(self, inputs): ...
    def release(self): ...
```

Providers wrap concrete models or external services. They never call other providers.

### Model

The actual model code lives inside each provider. Business logic never depends on a specific model.

## Data Flow

1. **Input**: video/audio loaded into `workspace/input/`.
2. **ASR**: speech-to-text produces a draft Timeline.
3. **Timeline**: forced alignment refines timings.
4. **Translation**: sentences translated to target language.
5. **Length Optimizer**: text adjusted to fit duration.
6. **Prosody**: emotion, speech rate, pauses annotated.
7. **TTS**: audio segments synthesized into `workspace/tts/`.
8. **Audio Post**: segments mixed and normalized.
9. **Face Animation**: lip-sync video generated into `workspace/face/`.
10. **Video Post**: final video muxed.
11. **Output**: artifacts delivered to `workspace/output/`.

## Frontend

The frontend is a Tauri + Vue + TypeScript application. It communicates with the Python backend exclusively through HTTP and WebSocket APIs. Tauri is only used for native file dialogs and local file access during development.

## Deployment

- Development: Tauri frontend + local Python API.
- Production Web: static frontend deployed to a web server + Python API on AI server.
- Production Desktop: Tauri app + Python API on AI server.

## Extensibility

Adding a new model: implement a new Provider and register it in `registry.py`.
Adding a new pipeline step: implement a new Engine and add it to the default steps.
Adding a new frontend capability: extend the API schemas and add new views.
