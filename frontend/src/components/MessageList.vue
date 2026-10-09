<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'

import type { ChatMessage, DetectionRecord } from '@/types'

import MessageItem from './MessageItem.vue'

const props = defineProps<{
  messages: ChatMessage[]
  detections: Record<string, DetectionRecord>
  renderMd: boolean
}>()

const emit = defineEmits<{ (e: 'open-detection', id: string): void }>()

const container = ref<HTMLElement | null>(null)

// 内容或条数变化时自动滚到底部
watch(
  () => [props.messages.length, props.messages.at(-1)?.content] as const,
  async () => {
    await nextTick()
    const el = container.value
    if (el) el.scrollTop = el.scrollHeight
  },
  { flush: 'post' },
)
</script>

<template>
  <div ref="container" class="msg-list">
    <el-empty v-if="messages.length === 0" description="开始新的对话吧" />
    <MessageItem
      v-for="message in messages"
      :key="message.id"
      :message="message"
      :detection="detections[message.id]"
      :render-md="renderMd"
      @open-detection="(id: string) => emit('open-detection', id)"
    />
  </div>
</template>
