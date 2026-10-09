import { describe, expect, it } from 'vitest'

import { renderMarkdown } from '@/utils/markdown'

describe('renderMarkdown', () => {
  it('renders basic markdown', () => {
    expect(renderMarkdown('# 标题')).toContain('<h1>')
  })

  it('escapes raw script tags', () => {
    const html = renderMarkdown('hello <script>alert(1)</script>')
    expect(html).not.toContain('<script>')
  })

  it('neutralizes dangerous raw HTML instead of emitting a live element', () => {
    const html = renderMarkdown('<img src=x onerror=alert(1)>')
    expect(html).not.toContain('<img')
  })
})
