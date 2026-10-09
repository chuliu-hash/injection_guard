import { describe, expect, it } from 'vitest'

import { parseSseBuffer } from '@/api/sse'

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
