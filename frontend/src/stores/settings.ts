import { defineStore } from 'pinia'
import { fetchChatDefaults } from '@/api/defense'
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
    /** 聊天模型 API 地址（请求级配置；留空使用后端默认）。 */
    chatApiUrl: '',
    /** 聊天模型 API Key：只在内存中保存，不进入 localStorage 或数据库。 */
    chatApiKey: '',
    /** 聊天模型名称。 */
    chatModel: '',
    /** 是否已经从后端载入过聊天默认值。 */
    chatSettingsInitialized: false,
    /** 服务端是否配置了默认 API Key（只返回布尔值）。 */
    chatServerKeyConfigured: false,
    /** 聊天生成温度（随请求体 temperature 发送）。 */
    chatTemperature: 0.7,
    /** 聊天单次生成 token 上限（随请求体 max_tokens 发送）。 */
    chatMaxTokens: 1024,
    /** 核采样阈值。 */
    chatTopP: 0.9,
    /** 频率惩罚（-2~2）。 */
    chatFrequencyPenalty: 0,
    /** 存在惩罚（-2~2）。 */
    chatPresencePenalty: 0,
    /** 固定采样种子；null 表示不固定。 */
    chatSeed: null as number | null,
    /** 输入判安全且 p_safe ≥ 此阈值时跳过输出检测（随请求体发送）。 */
    safeSkipThreshold: 0.5,
    /** 是否启用「高置信跳过输出检测」（随请求体发送）。 */
    skipOutputWhenHighConfidence: true,
  }),
  actions: {
    /** 首次进入应用时载入后端默认值；force=true 用于「恢复服务端默认」。 */
    async ensureChatDefaults(force = false): Promise<void> {
      if (this.chatSettingsInitialized && !force) return
      const defaults = await fetchChatDefaults()
      this.chatApiUrl = defaults.api_url
      this.chatModel = defaults.model
      this.chatServerKeyConfigured = defaults.api_key_configured
      this.chatTemperature = defaults.temperature
      this.chatMaxTokens = defaults.max_tokens
      this.chatTopP = defaults.top_p
      this.chatFrequencyPenalty = defaults.frequency_penalty
      this.chatPresencePenalty = defaults.presence_penalty
      this.chatSeed = defaults.seed
      if (force) this.chatApiKey = ''
      this.chatSettingsInitialized = true
    },
  },
  persist: {
    // API Key 不持久化：避免把模型凭据写入浏览器 localStorage。
    pick: [
      'useRag',
      'sensitivity',
      'renderMarkdown',
      'chatApiUrl',
      'chatModel',
      'chatSettingsInitialized',
      'chatTemperature',
      'chatMaxTokens',
      'chatTopP',
      'chatFrequencyPenalty',
      'chatPresencePenalty',
      'chatSeed',
      'safeSkipThreshold',
      'skipOutputWhenHighConfidence',
    ],
    // 旧版本持久化里 sensitivity 可能是 null（已移除的「跟随后端默认」），统一回落 low。
    afterHydrate: (ctx) => {
      if (!SENSITIVITY_LEVELS.includes(ctx.store.sensitivity)) ctx.store.sensitivity = 'low'
    },
  },
})
