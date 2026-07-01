import { defineStore } from 'pinia'
import { ref } from 'vue'
import { createTask, getTask, cancelTask, listProviders, connectProgress, type Task, type ProviderInfo, type ProgressEvent } from '@/api/tasks'

export const useTaskStore = defineStore('task', () => {
  const currentTask = ref<Task | null>(null)
  const providers = ref<ProviderInfo[]>([])
  const logs = ref<ProgressEvent[]>([])
  const socket = ref<WebSocket | null>(null)

  async function createNewTask(inputPath: string, configPath: string, resume: boolean): Promise<void> {
    const task = await createTask({ input_path: inputPath, config_path: configPath, resume })
    currentTask.value = task
    logs.value = []
    socket.value = connectProgress(task.task_id, (event) => {
      logs.value.push(event)
      if (event.status === 'completed' || event.status === 'failed') {
        refreshTask()
      }
    })
  }

  async function refreshTask(): Promise<void> {
    if (!currentTask.value) return
    currentTask.value = await getTask(currentTask.value.task_id)
  }

  async function cancelCurrentTask(): Promise<void> {
    if (!currentTask.value) return
    currentTask.value = await cancelTask(currentTask.value.task_id)
  }

  async function loadProviders(): Promise<void> {
    providers.value = await listProviders()
  }

  function closeSocket(): void {
    socket.value?.close()
    socket.value = null
  }

  return {
    currentTask,
    providers,
    logs,
    createNewTask,
    refreshTask,
    cancelCurrentTask,
    loadProviders,
    closeSocket,
  }
})
