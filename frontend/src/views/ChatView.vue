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
import { OUTPUT_BLOCKED_TEXT, resolveFinalPatch } from '@/utils/chatMessage'
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

onMounted(async () => {
  try {
    const param = route.params.sessionId
    const session = await store.activate(typeof param === 'string' ? param : undefined)
    if (route.params.sessionId !== session.id) void router.replace(`/chat/${session.id}`)
  } catch (error) {
    ElMessage.error(`加载会话失败：${error instanceof Error ? error.message : String(error)}`)
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

  const current = await store.ensureSession()
  const sessionId = current.id
  await store.appendMessage(sessionId, 'user', text)
  const assistant = await store.appendMessage(sessionId, 'assistant', '')

  const record: DetectionRecord = { isSafe: true, stoppedAt: null }
  let streamError: string | null = null
  // 服务端 done 事件里的完整回复（落库以此为准，流式增量只在本地）
  let modelResponse: string | null = null

  /** 本地累积的助手正文（流式 delta 只存内存）。 */
  const localContent = (): string =>
    store.sessions.find((s) => s.id === sessionId)?.messages.find((m) => m.id === assistant.id)
      ?.content ?? ''

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
          // 输出不安全：服务端已发出的 token 撤回不了，只能在此「检测结果一出来就替换」——
          // 立即用拦截文案覆盖气泡内容。可能是生成中途的增量检测命中（此时后面的 token
          // 已被服务端中断），也可能是生成结束后整段检测命中，两种情况的处理相同。
          if (!payload.is_safe) {
            record.stoppedAt = 'output_detection'
            void store.patchMessage(sessionId, assistant.id, { content: OUTPUT_BLOCKED_TEXT })
          }
        } else if (event === 'done') {
          const payload = data as DonePayload
          record.isSafe = payload.is_safe
          record.stoppedAt = payload.stopped_at
          modelResponse = payload.model_response
        } else if (event === 'error') {
          streamError = (data as { error: string }).error
        }
      },
      controller.signal,
    )

    await store.patchMessage(
      sessionId,
      assistant.id,
      resolveFinalPatch({
        streamError,
        stoppedAt: record.stoppedAt,
        localContent: localContent(),
        modelResponse,
      }),
    )
    if (streamError) ElMessage.error(streamError)
    await store.attachDetection(sessionId, assistant.id, record)
  } catch (error) {
    const aborted = controller?.signal.aborted === true
    const text2 = aborted ? '已停止生成' : error instanceof Error ? error.message : String(error)
    await store.patchMessage(sessionId, assistant.id, {
      content: localContent(),
      status: 'error',
      error: text2,
    })
    await store.attachDetection(sessionId, assistant.id, record)
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
