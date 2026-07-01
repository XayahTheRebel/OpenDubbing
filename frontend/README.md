# OpenDubbing Frontend

Tauri + Vue + TypeScript frontend for OpenDubbing.

## Development

```bash
npm install
npm run tauri:dev
```

## Build

```bash
npm run tauri:build
```

## Architecture

- `src/api/`: HTTP/WebSocket wrappers to the OpenDubbing API server.
- `src/stores/`: Pinia state management.
- `src/views/`: Page components.
- `src/components/`: Reusable UI components.
