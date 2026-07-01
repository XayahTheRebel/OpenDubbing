<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useTaskStore } from '@/stores/taskStore'

const store = useTaskStore()
const inputPath = ref('')
const configPath = ref('')
const resume = ref(false)

async function startTask(): Promise<void> {
  await store.createNewTask(inputPath.value, configPath.value, resume.value)
}

async function cancelTask(): Promise<void> {
  await store.cancelCurrentTask()
}

onMounted(() => {
  store.loadProviders()
})
</script>

<template>
  <div class="task-view">
    <section class="form">
      <h2>New Task</h2>
      <label>
        Input Path
        <input v-model="inputPath" type="text" placeholder="/path/to/video.mp4" />
      </label>
      <label>
        Config Path
        <input v-model="configPath" type="text" placeholder="/path/to/config.yaml" />
      </label>
      <label>
        <input v-model="resume" type="checkbox" />
        Resume from cache
      </label>
      <button @click="startTask" :disabled="!inputPath || !configPath">Start</button>
      <button v-if="store.currentTask?.status === 'running'" @click="cancelTask">Cancel</button>
    </section>

    <section v-if="store.currentTask" class="status">
      <h2>Status</h2>
      <p>ID: {{ store.currentTask.task_id }}</p>
      <p>Status: {{ store.currentTask.status }}</p>
      <p>Step: {{ store.currentTask.current_step || '-' }}</p>
      <progress :value="store.currentTask.progress" max="100">{{ store.currentTask.progress }}%</progress>
      <p v-if="store.currentTask.error" class="error">{{ store.currentTask.error }}</p>
      <p v-if="store.currentTask.output_path">Output: {{ store.currentTask.output_path }}</p>
    </section>

    <section v-if="store.logs.length > 0" class="logs">
      <h2>Logs</h2>
      <ul>
        <li v-for="(log, index) in store.logs" :key="index">
          [{{ log.step }}] {{ log.status }}
        </li>
      </ul>
    </section>

    <section class="providers">
      <h2>Providers</h2>
      <ul>
        <li v-for="provider in store.providers" :key="`${provider.kind}-${provider.name}`">
          {{ provider.kind }} / {{ provider.name }}
        </li>
      </ul>
    </section>
  </div>
</template>

<style scoped>
.task-view {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}
.form label {
  display: block;
  margin-bottom: 0.5rem;
}
.form input[type='text'] {
  width: 100%;
  padding: 0.5rem;
  margin-top: 0.25rem;
}
.status progress {
  width: 100%;
}
.error {
  color: #c00;
}
.logs ul {
  max-height: 300px;
  overflow-y: auto;
  background: #f5f5f5;
  padding: 0.5rem;
}
</style>
