/**
 * 助手消息的「终态落库」决策。
 *
 * 流式增量（`delta`）只改本地缓存，落库在结束时统一进行——这里集中决定该写什么，
 * 避免出现「刷新后助手回复空白」：正文必须以服务端 `done` 事件的 `model_response`
 * 为准写入 `content`，历史上这里只写了 `status`，导致回复正文从未落库。
 *
 * 服务端已按同一规则在生成结束时落库（见 `api/routes._AssistantPersistTarget`），
 * 前端这次 PATCH 是幂等兜底：内容一致，服务端落库失败或未带会话定位时仍能存住。
 */
import type { StoppedAt } from '@/types'

/** 输入检测拦截（未生成任何内容）。 */
export const BLOCKED_TEXT = '该消息未通过安全检测，已被拦截。'
/** 输出检测拦截（已生成部分内容，事后替换）。 */
export const OUTPUT_BLOCKED_TEXT = '该回复未通过输出安全检测，已被拦截。'

export interface FinalMessagePatch {
  content?: string
  status: 'done' | 'error'
  error?: string
}

export interface FinalPatchInput {
  /** 流式过程中的错误信息（HTTP 错误 / 中断文案）。 */
  streamError: string | null
  stoppedAt: StoppedAt
  /** 本地累积的正文（流式 delta）。 */
  localContent: string
  /** 服务端 `done` 事件里的完整回复；无则为 null。 */
  modelResponse: string | null
}

export function resolveFinalPatch({
  streamError,
  stoppedAt,
  localContent,
  modelResponse,
}: FinalPatchInput): FinalMessagePatch {
  if (streamError) {
    // 出错也要保留已生成的部分内容，而不是只留状态
    return { content: localContent, status: 'error', error: streamError }
  }
  if (stoppedAt === 'input_detection') {
    return { content: localContent || BLOCKED_TEXT, status: 'done' }
  }
  if (stoppedAt === 'output_detection') {
    // 气泡内容已在事件回调里替换为拦截文案，落库用同一文案，避免刷新后露出原文
    return { content: OUTPUT_BLOCKED_TEXT, status: 'done' }
  }
  // 正常完成 / 用户中断：以服务端完整回复为准，缺失时退回本地累积内容
  return { content: modelResponse ?? localContent, status: 'done' }
}
