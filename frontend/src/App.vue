<script setup lang="ts">
import { RouterView } from 'vue-router'
</script>

<template>
  <div class="app">
    <header>
      <h1>OpenDubbing</h1>
    </header>
    <main>
      <RouterView />
    </main>
  </div>
</template>

<style scoped>
.app {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  max-width: 960px;
  margin: 0 auto;
  padding: 1rem;
}
header h1 {
  margin: 0 0 1rem;
}
</style>
