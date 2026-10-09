import DOMPurify from 'dompurify'
import hljs from 'highlight.js'
import MarkdownIt from 'markdown-it'

const md = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
  highlight(code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return hljs.highlight(code, { language: lang }).value
      } catch {
        // 忽略高亮异常，回退为纯文本
      }
    }
    return ''
  },
})

/** 渲染 Markdown 并做 XSS 消毒（LLM 输出不可信）。 */
export function renderMarkdown(source: string): string {
  const raw = md.render(source ?? '')
  return DOMPurify.sanitize(raw)
}
