import { defineStore } from 'pinia'
import type { SensitivityLevel } from '@/types'

const SENSITIVITY_LEVELS: SensitivityLevel[] = ['low', 'medium', 'high']

export const useSettingsStore = defineStore('settings', {
  state: () => ({
    /**
     * RAG 增强开关（唯一实时状态：设置页与输入框开关共用，随请求体 use_rag 发送）。
     * 持久化，故刷新后保持上次选择。
     */
    useRag: false,
    /**
     * 输入检测敏感度（必选，随请求体 sensitivity 发送）。
     * 默认 low，与后端默认同档；无「跟随后端默认」这一档。
     */
    sensitivity: 'low' as SensitivityLevel,
    /** 是否把模型回复按 Markdown 渲染。 */
    renderMarkdown: true,
    /** 聊天生成温度（随请求体 temperature 发送）。 */
    chatTemperature: 0.7,
    /** 聊天单次生成 token 上限（随请求体 max_tokens 发送）。 */
    chatMaxTokens: 4096,
    /** 输入判安全且 p_safe ≥ 此阈值时跳过输出检测（随请求体发送）。 */
    safeSkipThreshold: 0.5,
    /** 是否启用「高置信跳过输出检测」（随请求体发送）。 */
    skipOutputWhenHighConfidence: true,
  }),
  persist: {
    // 旧版本持久化里 sensitivity 可能是 null（已移除的「跟随后端默认」），统一回落 low。
    afterHydrate: (ctx) => {
      if (!SENSITIVITY_LEVELS.includes(ctx.store.sensitivity)) ctx.store.sensitivity = 'low'
    },
  },
})
