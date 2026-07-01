export interface Task {
  task_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  current_step: string | null
  progress: number
  error: string | null
  output_path: string | null
}

export interface CreateTaskRequest {
  input_path: string
  config_path: string
  resume: boolean
}

export interface ProviderInfo {
  kind: string
  name: string
}

export interface ProgressEvent {
  task_id: string
  step: string
  status: string
  progress: number
  detail: Record<string, unknown>
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api/v1'

export async function createTask(request: CreateTaskRequest): Promise<Task> {
  const response = await fetch(`${API_BASE}/tasks`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!response.ok) {
    throw new Error(`Failed to create task: ${response.statusText}`)
  }
  return response.json()
}

export async function getTask(taskId: string): Promise<Task> {
  const response = await fetch(`${API_BASE}/tasks/${taskId}`)
  if (!response.ok) {
    throw new Error(`Failed to get task: ${response.statusText}`)
  }
  return response.json()
}

export async function cancelTask(taskId: string): Promise<Task> {
  const response = await fetch(`${API_BASE}/tasks/${taskId}/cancel`, {
    method: 'POST',
  })
  if (!response.ok) {
    throw new Error(`Failed to cancel task: ${response.statusText}`)
  }
  return response.json()
}

export async function listProviders(): Promise<ProviderInfo[]> {
  const response = await fetch(`${API_BASE}/providers`)
  if (!response.ok) {
    throw new Error(`Failed to list providers: ${response.statusText}`)
  }
  return response.json()
}

export function connectProgress(taskId: string, onMessage: (event: ProgressEvent) => void): WebSocket {
  const wsUrl = API_BASE.replace(/^http/, 'ws')
  const socket = new WebSocket(`${wsUrl}/ws/tasks/${taskId}`)
  socket.onmessage = (event) => {
    onMessage(JSON.parse(event.data))
  }
  return socket
}
