# OpenDubbing API

## Base URL

Development: `http://127.0.0.1:8000/api/v1`

## Endpoints

### Health

```http
GET /health
```

Response:

```json
{ "status": "ok" }
```

### List Providers

```http
GET /providers
```

Response:

```json
[
  { "kind": "tts", "name": "cosyvoice2" },
  { "kind": "asr", "name": "qwen3_asr" }
]
```

### Create Task

```http
POST /tasks
Content-Type: application/json

{
  "input_path": "/path/to/video.mp4",
  "config_path": "/path/to/config.yaml",
  "resume": false
}
```

Response:

```json
{
  "task_id": "uuid",
  "status": "running",
  "current_step": null,
  "progress": 0,
  "error": null,
  "output_path": null
}
```

### Get Task

```http
GET /tasks/{task_id}
```

### Cancel Task

```http
POST /tasks/{task_id}/cancel
```

## WebSocket

Connect to:

```
ws://127.0.0.1:8000/api/v1/ws/tasks/{task_id}
```

Progress events:

```json
{
  "task_id": "uuid",
  "step": "asr",
  "status": "started",
  "progress": 0,
  "detail": {}
}
```

Status values: `started`, `completed`, `cached`, `skipped`, `failed`.
