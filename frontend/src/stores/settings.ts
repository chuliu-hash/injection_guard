import { defineStore } from 'pinia'

export const useSettingsStore = defineStore('settings', {
  state: () => ({
    /** 输入框 RAG 开关的初始值。 */
    defaultUseRag: false,
    /** 是否把模型回复按 Markdown 渲染。 */
    renderMarkdown: true,
  }),
  persist: true,
})
