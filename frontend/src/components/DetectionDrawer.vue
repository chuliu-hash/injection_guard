<script setup lang="ts">
import { computed, ref, watch } from 'vue'

import { useSessionStore } from '@/stores/session'
import type { ChatMessage, DetectionRecord } from '@/types'
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

/** 该助手回合对应的用户输入：向前找最近的一条 user 消息。
 *  检测记录针对的是**用户输入**，预览理应显示它，而非模型回答。
 *  （ChatView.send 总是先追加 user 再追加 assistant，故必然能配到。） */
function userInputBefore(messages: ChatMessage[], index: number): string {
  for (let i = index - 1; i >= 0; i -= 1) {
    if (messages[i].role === 'user') return messages[i].content
  }
  return ''
}

const items = computed<RecordItem[]>(() => {
  const session = store.sessions.find((s) => s.id === props.sessionId)
  if (!session) return []
  const messages = session.messages
  return messages
    .map((message, index) => ({ message, index }))
    .filter(({ message }) => message.role === 'assistant' && session.detections[message.id])
    .map(({ message, index }) => ({
      id: message.id,
      createdAt: message.createdAt,
      preview: truncate(userInputBefore(messages, index), 40),
      record: session.detections[message.id],
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
