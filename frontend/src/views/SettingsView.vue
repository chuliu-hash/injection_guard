<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'

import { testServices } from '@/api/defense'
import { useSessionStore } from '@/stores/session'
import { useSettingsStore } from '@/stores/settings'
import type { ServiceKey, ServiceProbeResult } from '@/types'

const settings = useSettingsStore()
const session = useSessionStore()

// ── 服务连通性 ──────────────────────────────────────────────
/** 固定展示顺序（与后端 SERVICE_KEYS 一致）。 */
const SERVICE_ORDER: ServiceKey[] = ['input', 'output', 'rag', 'chat']

/** 覆盖本地大模型首次懒加载；服务不可达时连接会立即失败，不会白等满 60 秒。 */
const PROBE_TIMEOUT_SECONDS = 60

const probing = ref(false)
const probeResults = ref<Partial<Record<ServiceKey, ServiceProbeResult>> | null>(null)

async function runProbe(): Promise<void> {
  probing.value = true
  try {
    const report = await testServices({
      timeout: PROBE_TIMEOUT_SECONDS,
      chat_api_url: settings.chatApiUrl.trim() || undefined,
      chat_api_key: settings.chatApiKey.trim() || undefined,
      chat_model: settings.chatModel.trim() || undefined,
    })
    probeResults.value = report.services
    if (report.ok) ElMessage.success('四个服务均连通')
    else ElMessage.warning('部分服务不可用，详见列表')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : String(error))
  } finally {
    probing.value = false
  }
}

function probeOf(key: ServiceKey): ServiceProbeResult | undefined {
  return probeResults.value?.[key]
}

async function loadChatDefaults(force = false): Promise<void> {
  try {
    await settings.ensureChatDefaults(force)
    if (force) ElMessage.success('已恢复服务端默认聊天配置')
  } catch (error) {
    ElMessage.error(`加载聊天默认配置失败：${error instanceof Error ? error.message : String(error)}`)
  }
}

onMounted(() => {
  void loadChatDefaults()
})

async function clearAll(): Promise<void> {
  try {
    await ElMessageBox.confirm('将清空所有会话与检测记录，且不可恢复，确定继续？', '清空数据', {
      type: 'warning',
      confirmButtonText: '清空',
      cancelButtonText: '取消',
    })
    await session.clearAll()
    ElMessage.success('已清空')
  } catch {
    // 用户取消
  }
}
</script>

<template>
  <div class="settings">
    <h3>设置</h3>
    <el-form label-width="150px">
      <el-form-item label="服务连通性">
        <el-button type="primary" :loading="probing" @click="runProbe">测试连接</el-button>
        <span class="hint">对输入/输出/RAG/聊天四个模型各发一次最小请求</span>
        <div v-if="probeResults" class="probe-list">
          <div v-for="key in SERVICE_ORDER" :key="key" class="probe-row">
            <template v-if="probeOf(key)">
              <el-tag :type="probeOf(key)!.ok ? 'success' : 'danger'" size="small" disable-transitions>
                {{ probeOf(key)!.ok ? '成功' : '失败' }}
              </el-tag>
              <span class="probe-row__label">{{ probeOf(key)!.label }}</span>
              <span v-if="probeOf(key)!.ok" class="probe-row__meta">
                {{ probeOf(key)!.latency_ms }} ms · {{ probeOf(key)!.model }}
              </span>
              <span v-else class="probe-row__error">{{ probeOf(key)!.error }}</span>
            </template>
          </div>
        </div>
      </el-form-item>

      <el-form-item label="RAG 增强">
        <el-switch v-model="settings.useRag" />
        <span class="hint">
          是否对输入进行安全检索增强
        </span>
      </el-form-item>

      <el-form-item label="输入检测敏感度">
        <el-select v-model="settings.sensitivity" style="width: 200px">
          <el-option label="低（低误报，可能漏检）" value="low" />
          <el-option label="中（平衡）" value="medium" />
          <el-option label="高（严检出，可能误报）" value="high" />
        </el-select>
        <span class="hint">越高越倾向判为不安全</span>
      </el-form-item>

      <el-form-item label="Markdown 渲染">
        <el-switch v-model="settings.renderMarkdown" />
        <span class="hint">关闭后以纯文本显示模型回复</span>
      </el-form-item>

      <el-divider content-position="left">聊天模型</el-divider>

      <el-form-item label="API 地址">
        <el-input
          v-model="settings.chatApiUrl"
          placeholder="https://api.openai.com/v1"
          style="max-width: 460px"
        />
        <span class="hint">OpenAI 兼容的 base URL；留空使用后端默认值</span>
      </el-form-item>

      <el-form-item label="API Key">
        <el-input
          v-model="settings.chatApiKey"
          type="password"
          show-password
          autocomplete="off"
          :placeholder="settings.chatServerKeyConfigured ? '留空使用服务端已配置的 Key' : '输入模型 API Key'"
          style="max-width: 460px"
        />
        <span class="hint">仅保存在当前页面内存，不写 localStorage / 数据库；刷新后需重新输入</span>
      </el-form-item>

      <el-form-item label="模型名称">
        <el-input
          v-model="settings.chatModel"
          placeholder="例如 gpt-4.1-mini / deepseek-chat"
          style="max-width: 460px"
        />
        <span class="hint">留空使用后端默认模型</span>
      </el-form-item>

      <el-form-item label="服务端默认">
        <el-button @click="loadChatDefaults(true)">恢复服务端默认</el-button>
        <span class="hint">后端 env 只作为默认值，运行中修改 env 需重启后重新载入</span>
      </el-form-item>

      <el-divider content-position="left">生成参数</el-divider>

      <el-form-item label="生成温度">
        <el-slider
          v-model="settings.chatTemperature"
          :min="0"
          :max="2"
          :step="0.1"
          show-input
          style="max-width: 320px"
        />
        <span class="hint">越高越随机（0~2）；随请求发送给聊天模型</span>
      </el-form-item>

      <el-form-item label="最大生成长度">
        <el-input-number v-model="settings.chatMaxTokens" :min="1" :max="32768" :step="256" />
        <span class="hint">单次回复的 token 上限</span>
      </el-form-item>

      <el-form-item label="Top P">
        <el-slider
          v-model="settings.chatTopP"
          :min="0.05"
          :max="1"
          :step="0.05"
          show-input
          style="max-width: 320px"
        />
        <span class="hint">核采样范围；越低越保守（0.05~1）</span>
      </el-form-item>

      <el-form-item label="频率惩罚">
        <el-slider
          v-model="settings.chatFrequencyPenalty"
          :min="-2"
          :max="2"
          :step="0.1"
          show-input
          style="max-width: 320px"
        />
        <span class="hint">降低重复用词；本地模型以 repetition penalty 近似实现</span>
      </el-form-item>

      <el-form-item label="存在惩罚">
        <el-slider
          v-model="settings.chatPresencePenalty"
          :min="-2"
          :max="2"
          :step="0.1"
          show-input
          style="max-width: 320px"
        />
        <span class="hint">鼓励模型讨论新话题；本地模型以 repetition penalty 近似实现</span>
      </el-form-item>

      <el-form-item label="随机种子">
        <el-input-number
          v-model="settings.chatSeed"
          :min="0"
          :max="2147483647"
          :step="1"
          placeholder="留空表示不固定"
        />
        <span class="hint">固定后可提高同参数下输出复现性</span>
      </el-form-item>

      <el-form-item label="跳过高置信输出检测">
        <el-switch v-model="settings.skipOutputWhenHighConfidence" />
        <span class="hint">
          开启后，输入判安全且置信度达标时不再做输出检测
        </span>
      </el-form-item>

      <el-form-item label="高置信阈值">
        <el-slider
          v-model="settings.safeSkipThreshold"
          :min="0"
          :max="1"
          :step="0.05"
          :disabled="!settings.skipOutputWhenHighConfidence"
          show-input
          style="max-width: 320px"
        />
        <span class="hint">安全置信度达到该值即跳过输出检测（仅在上一项开启时生效）</span>
      </el-form-item>

      <el-form-item label="会话数据">
        <el-button type="danger" plain @click="clearAll">清空所有会话</el-button>
        <span class="hint">共 {{ session.sessions.length }} 个会话（保存在服务端数据库，按用户隔离）</span>
      </el-form-item>
    </el-form>
  </div>
</template>
