<script setup lang="ts">
import { computed, ref } from 'vue'

const props = defineProps<{
  sending: boolean
  useRag: boolean
}>()

const emit = defineEmits<{
  (e: 'send', text: string): void
  (e: 'stop'): void
  (e: 'update:useRag', value: boolean): void
}>()

const text = ref('')
const canSend = computed(() => text.value.trim().length > 0 && !props.sending)

function submit(): void {
  if (!canSend.value) return
  emit('send', text.value)
  text.value = ''
}

function onKeydown(event: KeyboardEvent): void {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault()
    submit()
  }
}
</script>

<template>
  <div class="composer">
    <div class="composer__toolbar">
      <el-switch
        :model-value="useRag"
        inline-prompt
        active-text="RAG 开"
        inactive-text="RAG 关"
        @update:model-value="(value: string | number | boolean) => emit('update:useRag', Boolean(value))"
      />
      <span style="font-size: 12px; color: var(--el-text-color-secondary)">
        开启后先检索安全示例再生成回复
      </span>
    </div>

    <el-input
      v-model="text"
      type="textarea"
      :rows="3"
      resize="none"
      :disabled="sending"
      placeholder="输入消息，Enter 发送，Shift + Enter 换行"
      @keydown="onKeydown"
    />

    <div class="composer__actions">
      <el-button v-if="sending" type="danger" plain @click="emit('stop')">停止生成</el-button>
      <el-button v-else type="primary" :disabled="!canSend" @click="submit">发送</el-button>
    </div>
  </div>
</template>
