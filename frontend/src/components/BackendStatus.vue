<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

import { fetchHealth } from '@/api/defense'

const online = ref<boolean | null>(null)
const detail = ref('正在检测后端…')
let timer: number | undefined

async function poll(): Promise<void> {
  try {
    const health = await fetchHealth()
    online.value = true
    detail.value = `输入/输出/聊天并发 ${health.input_max_concurrency}/${health.output_max_concurrency}/${health.chat_max_concurrency} · RAG ${
      health.enable_rag ? '开' : '关'
    }`
  } catch {
    online.value = false
    detail.value = '后端不可达（请确认防御 API 已启动）'
  }
}

onMounted(() => {
  void poll()
  timer = window.setInterval(poll, 30000)
})

onBeforeUnmount(() => {
  if (timer !== undefined) window.clearInterval(timer)
})

const statusText = computed(() =>
  online.value === null ? '检测中' : online.value ? '在线' : '离线',
)
const statusType = computed(() => (online.value === false ? 'danger' : 'success'))
</script>

<template>
  <el-tooltip :content="detail" placement="bottom">
    <span class="backend-status">
      <el-badge is-dot :type="statusType" />
      <span>后端 {{ statusText }}</span>
    </span>
  </el-tooltip>
</template>
