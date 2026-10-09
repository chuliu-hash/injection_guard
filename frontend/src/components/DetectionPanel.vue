<script setup lang="ts">
import { computed } from 'vue'

import type { DetectionRecord } from '@/types'
import { percent } from '@/utils/format'

import RiskTag from './RiskTag.vue'

const props = defineProps<{ record: DetectionRecord }>()

const input = computed(() => props.record.inputDetection)
const rag = computed(() => props.record.rag)
const output = computed(() => props.record.outputDetection)

const stoppedLabel = computed(() => {
  switch (props.record.stoppedAt) {
    case 'input_detection':
      return '输入检测拦截'
    case 'output_skipped':
      return '高置信跳过输出检测'
    default:
      return '全流程完成'
  }
})

const passedRules = computed(() => {
  const total = output.value?.total_rules ?? 0
  const failed = output.value?.failed_rules?.length ?? 0
  return total - failed
})
</script>

<template>
  <div class="detection-panel">
    <el-descriptions :column="1" size="small" border>
      <el-descriptions-item label="最终判定">
        <RiskTag :safe="record.isSafe" />
      </el-descriptions-item>
      <el-descriptions-item label="终止阶段">{{ stoppedLabel }}</el-descriptions-item>
    </el-descriptions>

    <template v-if="input">
      <h4>输入检测</h4>
      <el-descriptions :column="1" size="small" border>
        <el-descriptions-item label="风险等级">{{ input.risk_level }}</el-descriptions-item>
        <el-descriptions-item label="风险类别">{{ input.category }}</el-descriptions-item>
        <el-descriptions-item label="决策方式">{{ input.decision_by }}</el-descriptions-item>
        <el-descriptions-item label="置信度">{{ percent(input.confidence) }}</el-descriptions-item>
        <el-descriptions-item label="P(安全)">
          <el-progress
            :percentage="Math.round((input.p_safe ?? 0) * 100)"
            :stroke-width="10"
            :show-text="false"
          />
          <span class="detection-panel__num">{{ percent(input.p_safe) }}</span>
        </el-descriptions-item>
        <el-descriptions-item label="P(攻击)">
          <el-progress
            :percentage="Math.round((input.p_attack ?? 0) * 100)"
            :stroke-width="10"
            status="exception"
            :show-text="false"
          />
          <span class="detection-panel__num">{{ percent(input.p_attack) }}</span>
        </el-descriptions-item>
      </el-descriptions>
    </template>

    <template v-if="rag">
      <h4>RAG 增强</h4>
      <el-descriptions :column="1" size="small" border>
        <el-descriptions-item label="是否启用">
          <el-tag size="small" :type="rag.enabled ? 'success' : 'info'" effect="light">
            {{ rag.enabled ? '已启用' : '未启用' }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item v-if="rag.retrieved_contexts?.length" label="命中示例">
          {{ rag.retrieved_contexts.length }} 条
        </el-descriptions-item>
      </el-descriptions>

      <el-collapse v-if="rag.final_prompt">
        <el-collapse-item title="final_prompt（送入聊天接口的实际提示词）" name="fp">
          <pre class="fp">{{ rag.final_prompt }}</pre>
        </el-collapse-item>
      </el-collapse>

      <div v-if="rag.retrieved_contexts?.length" class="ctx">
        <div class="ctx__label">检索到的安全示例</div>
        <el-card
          v-for="(ctx, index) in rag.retrieved_contexts"
          :key="index"
          shadow="never"
          class="ctx__card"
        >
          <div class="ctx__q">Q: {{ ctx.prompt }}</div>
          <div class="ctx__a">A: {{ ctx.response }}</div>
        </el-card>
      </div>
    </template>

    <template v-if="output">
      <h4>输出检测</h4>
      <el-descriptions :column="1" size="small" border>
        <el-descriptions-item label="是否安全">
          <RiskTag :safe="output.is_safe" />
        </el-descriptions-item>
        <el-descriptions-item label="规则通过">
          {{ passedRules }} / {{ output.total_rules }}
        </el-descriptions-item>
        <el-descriptions-item v-if="output.failed_rules?.length" label="未通过规则">
          <el-tag
            v-for="rule in output.failed_rules"
            :key="rule"
            type="danger"
            size="small"
            class="mr"
            effect="light"
          >
            {{ rule }}
          </el-tag>
        </el-descriptions-item>
      </el-descriptions>
    </template>
  </div>
</template>

<style scoped>
.detection-panel__num {
  margin-left: 8px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>
