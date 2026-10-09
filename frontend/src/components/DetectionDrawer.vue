<script setup lang="ts">
import { computed, ref, watch } from 'vue'

import { useSessionStore } from '@/stores/session'
import type { DetectionRecord } from '@/types'
import { formatTime, truncate } from '@/utils/format'

import DetectionPanel from './DetectionPanel.vue'

const props = defineProps<{
  modelValue: boolean
  sessionId: string | null
  focusMessageId?: string | null
}>()

const emit = defineEmits<{ (e: 'update:modelValue', value: boolean): void }>()

const store = useSessionStore()
const active = ref<string[]>([])

interface RecordItem {
  id: string
  createdAt: number
  preview: string
  record: DetectionRecord
}

const items = computed<RecordItem[]>(() => {
  const session = store.sessions.find((s) => s.id === props.sessionId)
  if (!session) return []
  return session.messages
    .filter((m) => m.role === 'assistant' && session.detections[m.id])
    .map((m) => ({
      id: m.id,
      createdAt: m.createdAt,
      preview: truncate(m.content, 40),
      record: session.detections[m.id],
    }))
    .reverse()
})

watch(
  () => [props.modelValue, props.focusMessageId] as const,
  ([visible, focus]) => {
    if (visible && focus) active.value = [focus]
  },
  { immediate: true },
)
</script>

<template>
  <el-drawer
    :model-value="modelValue"
    title="检测记录"
    size="540px"
    @update:model-value="(value: boolean) => emit('update:modelValue', value)"
  >
    <el-empty v-if="items.length === 0" description="本会话暂无检测记录" />
    <el-collapse v-else v-model="active">
      <el-collapse-item v-for="item in items" :key="item.id" :name="item.id">
        <template #title>
          <div class="rec-title">
            <el-tag :type="item.record.isSafe ? 'success' : 'danger'" size="small" effect="light">
              {{ item.record.isSafe ? '安全' : '不安全' }}
            </el-tag>
            <span class="rec-title__text">{{ item.preview || '（无内容）' }}</span>
            <span class="rec-title__time">{{ formatTime(item.createdAt) }}</span>
          </div>
        </template>
        <DetectionPanel :record="item.record" />
      </el-collapse-item>
    </el-collapse>
  </el-drawer>
</template>
