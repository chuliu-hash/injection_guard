import { afterEach, describe, expect, it, vi } from 'vitest'

import { getAccessToken, registerRefreshHandler, setAccessToken } from '@/api/authToken'
import { parseSseBuffer, postSse } from '@/api/sse'

afterEach(() => {
  vi.unstubAllGlobals()
  setAccessToken(null)
  registerRefreshHandler(null)
})

describe('postSse', () => {
  it('attaches the bearer token so the streamed endpoint is authorized', async () => {
    setAccessToken('tok-1')
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response('event: done\ndata: {"is_safe":true}\n\n', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    const events: Array<[string, unknown]> = []
    await postSse({
      url: '/api/chat_stream',
      body: { user_input: 'hi' },
      onEvent: (event, data) => events.push([event, data]),
    })

    const [, init] = fetchMock.mock.calls[0]
    expect((init.headers as Record<string, string>).Authorization).toBe('Bearer tok-1')
    expect(events).toEqual([['done', { is_safe: true }]])
  })

  it('refreshes once and retries when the stream returns 401', async () => {
    setAccessToken('expired')
    registerRefreshHandler(async () => {
      setAccessToken('fresh')
      return true
    })
    const unauthorized = new Response(JSON.stringify({ error: '未认证' }), { status: 401 })
    const ok = new Response('event: done\ndata: {}\n\n', { status: 200 })
    const fetchMock = vi.fn().mockResolvedValueOnce(unauthorized).mockResolvedValueOnce(ok)
    vi.stubGlobal('fetch', fetchMock)

    await postSse({ url: '/api/chat_stream', body: {}, onEvent: () => {} })

    expect(fetchMock).toHaveBeenCalledTimes(2)
    const [, retryInit] = fetchMock.mock.calls[1]
    expect((retryInit.headers as Record<string, string>).Authorization).toBe('Bearer fresh')
    expect(getAccessToken()).toBe('fresh')
  })
})

describe('parseSseBuffer', () => {
  it('parses complete frames and leaves nothing in rest', () => {
    const { events, rest } = parseSseBuffer('event: a\ndata: {"x":1}\n\nevent: b\ndata: 2\n\n')
    expect(rest).toBe('')
    expect(events).toEqual([
      { event: 'a', data: '{"x":1}' },
      { event: 'b', data: '2' },
    ])
  })

  it('keeps an incomplete tail in rest', () => {
    const { events, rest } = parseSseBuffer('event: a\ndata: 1\n\nevent: b\ndata: par')
    expect(events).toEqual([{ event: 'a', data: '1' }])
    expect(rest).toBe('event: b\ndata: par')
  })

  it('ignores heartbeat comments and joins multi-line data', () => {
    const { events } = parseSseBuffer(': ping\n\nevent: a\ndata: l1\ndata: l2\n\n')
    expect(events).toEqual([{ event: 'a', data: 'l1\nl2' }])
  })

  it('defaults the event name to message', () => {
    const { events } = parseSseBuffer('data: hi\n\n')
    expect(events).toEqual([{ event: 'message', data: 'hi' }])
  })

  it('supports CRLF frame separators', () => {
    const { events } = parseSseBuffer('event: a\r\ndata: 1\r\n\r\n')
    expect(events).toEqual([{ event: 'a', data: '1' }])
  })
})
