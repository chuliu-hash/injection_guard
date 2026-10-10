<script setup lang="ts">
import { computed } from 'vue'
import { MagicStick, User, View, WarningFilled } from '@element-plus/icons-vue'

import type { ChatMessage, DetectionRecord } from '@/types'
import { renderMarkdown } from '@/utils/markdown'

const props = defineProps<{
  message: ChatMessage
  detection?: DetectionRecord
  renderMd: boolean
}>()

const emit = defineEmits<{ (e: 'open-detection', id: string): void }>()

const isUser = computed(() => props.message.role === 'user')
const isAssistant = computed(() => props.message.role === 'assistant')
const renderedHtml = computed(() =>
  props.renderMd && isAssistant.value ? renderMarkdown(props.message.content) : '',
)
// 输入检测拦截（未生成）与输出检测拦截（已生成、事后替换）都用拦截样式
const blocked = computed(
  () =>
    props.detection?.stoppedAt === 'input_detection' ||
    props.detection?.stoppedAt === 'output_detection',
)
</script>

<template>
  <div class="msg" :class="isUser ? 'msg--user' : 'msg--assistant'">
    <el-avatar :size="32">
      <el-icon><component :is="isUser ? User : MagicStick" /></el-icon>
    </el-avatar>

    <div class="msg__body">
      <div class="msg__bubble" :class="{ 'msg__bubble--blocked': blocked }">
        <div v-if="renderedHtml" class="md" v-html="renderedHtml" />
        <template v-else>{{ message.content }}</template>
        <span v-if="message.status === 'streaming'" class="cursor">▍</span>
      </div>

      <div v-if="message.status === 'error'" class="msg__error">
        <el-icon><WarningFilled /></el-icon>
        <span>{{ message.error }}</span>
      </div>

      <div class="msg__meta">
        <el-button
          v-if="detection"
          link
          type="info"
          size="small"
          @click="emit('open-detection', message.id)"
        >
          <el-icon><View /></el-icon>
          <span style="margin-left: 4px">检测详情</span>
        </el-button>
      </div>
    </div>
  </div>
</template>
