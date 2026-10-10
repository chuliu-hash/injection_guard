<script setup lang="ts">
import { ref } from 'vue'
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

const probing = ref(false)
const probeResults = ref<Partial<Record<ServiceKey, ServiceProbeResult>> | null>(null)

async function runProbe(): Promise<void> {
  probing.value = true
  try {
    const report = await testServices()
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

async function clearAll(): Promise<void> {
  try {
    await ElMessageBox.confirm('将清空所有会话与检测记录，且不可恢复，确定继续？', '清空数据', {
      type: 'warning',
      confirmButtonText: '清空',
      cancelButtonText: '取消',
    })
    session.clearAll()
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
        <span class="hint">共 {{ session.sessions.length }} 个会话（保存在浏览器本地）</span>
      </el-form-item>
    </el-form>
  </div>
</template>
