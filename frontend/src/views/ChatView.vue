<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Document } from '@element-plus/icons-vue'

import { chatStream } from '@/api/defense'
import ChatComposer from '@/components/ChatComposer.vue'
import DetectionDrawer from '@/components/DetectionDrawer.vue'
import MessageList from '@/components/MessageList.vue'
import { useSessionStore } from '@/stores/session'
import { useSettingsStore } from '@/stores/settings'
import type {
  DetectionRecord,
  DonePayload,
  InputDetection,
  OutputDetection,
  RagResult,
} from '@/types'

const store = useSessionStore()
const settings = useSettingsStore()
const route = useRoute()
const router = useRouter()

const sending = ref(false)
const drawerVisible = ref(false)
const focusMessageId = ref<string | null>(null)
let controller: AbortController | null = null

const session = computed(() => store.current)
const messages = computed(() => session.value?.messages ?? [])
const detections = computed(() => session.value?.detections ?? {})

const BLOCKED_TEXT = '该消息未通过安全检测，已被拦截。'

onMounted(() => {
  const param = route.params.sessionId
  if (typeof param === 'string' && store.sessions.some((s) => s.id === param)) {
    store.selectSession(param)
  } else {
    store.ensureSession()
    if (store.currentId) void router.replace(`/chat/${store.currentId}`)
  }
})

watch(
  () => store.currentId,
  (id) => {
    if (id && route.params.sessionId !== id) void router.replace(`/chat/${id}`)
  },
)

function openDetection(id?: string): void {
  focusMessageId.value = id ?? null
  drawerVisible.value = true
}

function stop(): void {
  controller?.abort()
}

async function send(rawText: string): Promise<void> {
  const text = rawText.trim()
  if (!text || sending.value) return

  const current = store.ensureSession()
  const sessionId = current.id
  store.appendMessage(sessionId, 'user', text)
  const assistant = store.appendMessage(sessionId, 'assistant', '')

  const record: DetectionRecord = { isSafe: true, stoppedAt: null }
  let streamError: string | null = null

  sending.value = true
  controller = new AbortController()

  try {
    await chatStream(
      {
        user_input: text,
        use_rag: settings.useRag,
        sensitivity: settings.sensitivity,
        temperature: settings.chatTemperature,
        max_tokens: settings.chatMaxTokens,
        safe_skip_threshold: settings.safeSkipThreshold,
        skip_output_when_high_confidence: settings.skipOutputWhenHighConfidence,
      },
      (event, data) => {
        if (event === 'input_detection') {
          const payload = data as InputDetection
          record.inputDetection = payload
          record.isSafe = payload.is_safe
        } else if (event === 'rag') {
          record.rag = data as RagResult
        } else if (event === 'delta') {
          store.appendMessageContent(sessionId, assistant.id, (data as { content: string }).content)
        } else if (event === 'output_detection') {
          const payload = data as OutputDetection
          record.outputDetection = payload
          record.isSafe = payload.is_safe
        } else if (event === 'done') {
          const payload = data as DonePayload
          record.isSafe = payload.is_safe
          record.stoppedAt = payload.stopped_at
        } else if (event === 'error') {
          streamError = (data as { error: string }).error
        }
      },
      controller.signal,
    )

    const message = store.sessions
      .find((s) => s.id === sessionId)
      ?.messages.find((m) => m.id === assistant.id)

    if (streamError) {
      store.patchMessage(sessionId, assistant.id, { status: 'error', error: streamError })
      ElMessage.error(streamError)
    } else if (record.stoppedAt === 'input_detection' && !message?.content) {
      store.patchMessage(sessionId, assistant.id, { content: BLOCKED_TEXT, status: 'done' })
    } else {
      store.patchMessage(sessionId, assistant.id, { status: 'done' })
    }
    store.attachDetection(sessionId, assistant.id, record)
  } catch (error) {
    const aborted = controller?.signal.aborted === true
    const text2 = aborted ? '已停止生成' : error instanceof Error ? error.message : String(error)
    store.patchMessage(sessionId, assistant.id, { status: 'error', error: text2 })
    store.attachDetection(sessionId, assistant.id, record)
    if (!aborted) ElMessage.error(text2)
  } finally {
    sending.value = false
    controller = null
  }
}
</script>

<template>
  <div class="chat">
    <div class="chat__bar">
      <span class="chat__name">{{ session?.title ?? '会话' }}</span>
      <el-button size="small" :icon="Document" @click="openDetection()">检测记录</el-button>
    </div>

    <MessageList
      class="chat__list"
      :messages="messages"
      :detections="detections"
      :render-md="settings.renderMarkdown"
      @open-detection="openDetection"
    />

    <ChatComposer
      :sending="sending"
      :use-rag="settings.useRag"
      @send="send"
      @stop="stop"
      @update:use-rag="(value: boolean) => (settings.useRag = value)"
    />

    <DetectionDrawer
      v-model="drawerVisible"
      :session-id="store.currentId"
      :focus-message-id="focusMessageId"
    />
  </div>
</template>
