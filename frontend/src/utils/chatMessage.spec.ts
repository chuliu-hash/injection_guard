import { describe, expect, it } from 'vitest'

import { BLOCKED_TEXT, OUTPUT_BLOCKED_TEXT, resolveFinalPatch } from '@/utils/chatMessage'

describe('resolveFinalPatch', () => {
  it('正常完成时以服务端 done 的完整回复落库', () => {
    expect(
      resolveFinalPatch({
        streamError: null,
        stoppedAt: null,
        localContent: '你好',
        modelResponse: '你好！有什么可以帮你？',
      }),
    ).toEqual({ content: '你好！有什么可以帮你？', status: 'done' })
  })

  it('缺少 model_response 时退回本地累积内容', () => {
    expect(
      resolveFinalPatch({
        streamError: null,
        stoppedAt: null,
        localContent: '已生成一半',
        modelResponse: null,
      }),
    ).toEqual({ content: '已生成一半', status: 'done' })
  })

  it('输入检测拦截：无内容时写入拦截文案', () => {
    expect(
      resolveFinalPatch({
        streamError: null,
        stoppedAt: 'input_detection',
        localContent: '',
        modelResponse: null,
      }),
    ).toEqual({ content: BLOCKED_TEXT, status: 'done' })
  })

  it('输出检测拦截：落库拦截文案而不是模型原文', () => {
    expect(
      resolveFinalPatch({
        streamError: null,
        stoppedAt: 'output_detection',
        localContent: '危险原文',
        modelResponse: '危险原文',
      }),
    ).toEqual({ content: OUTPUT_BLOCKED_TEXT, status: 'done' })
  })

  it('流式出错：保留已生成内容并记录错误', () => {
    expect(
      resolveFinalPatch({
        streamError: '请求失败（HTTP 401）',
        stoppedAt: null,
        localContent: '部分内容',
        modelResponse: null,
      }),
    ).toEqual({ content: '部分内容', status: 'error', error: '请求失败（HTTP 401）' })
  })
})
