<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

import { fetchHealth } from '@/api/defense'
import { useSettingsStore } from '@/stores/settings'
import type { HealthInfo } from '@/types'

const settings = useSettingsStore()

const online = ref<boolean | null>(null)
const health = ref<HealthInfo | null>(null)
let timer: number | undefined

async function poll(): Promise<void> {
  try {
    health.value = await fetchHealth()
    online.value = true
  } catch {
    health.value = null
    online.value = false
  }
}

/** 提示文本：并发数取自后端 /health；RAG 与敏感度取**前端当前实际会发送的值**
 *  （均来自共享设置），随设置变化即时更新。 */
const detail = computed(() => {
  if (online.value === false) return '后端不可达（请确认防御 API 已启动）'
  if (!health.value) return '正在检测后端…'
  const h = health.value
  return `输入/输出/聊天并发 ${h.input_max_concurrency}/${h.output_max_concurrency}/${h.chat_max_concurrency} · RAG ${
    settings.useRag ? '开' : '关'
  } · 敏感度 ${settings.sensitivity}`
})

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
