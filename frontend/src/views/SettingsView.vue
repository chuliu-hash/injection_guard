<script setup lang="ts">
import { computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'

import { useSessionStore } from '@/stores/session'
import { useSettingsStore } from '@/stores/settings'

const settings = useSettingsStore()
const session = useSessionStore()

const apiBase = computed(() => import.meta.env.VITE_API_BASE_URL || '（同源，由后端托管）')

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
      <el-form-item label="后端地址">
        <el-tag type="info">{{ apiBase }}</el-tag>
        <span class="hint">由构建期 VITE_API_BASE_URL 决定；同源部署无需配置</span>
      </el-form-item>

      <el-form-item label="默认开启 RAG">
        <el-switch v-model="settings.defaultUseRag" />
        <span class="hint">仅作为输入框开关的初始值</span>
      </el-form-item>

      <el-form-item label="Markdown 渲染">
        <el-switch v-model="settings.renderMarkdown" />
        <span class="hint">关闭后以纯文本显示模型回复</span>
      </el-form-item>

      <el-form-item label="会话数据">
        <el-button type="danger" plain @click="clearAll">清空所有会话</el-button>
        <span class="hint">共 {{ session.sessions.length }} 个会话（保存在浏览器本地）</span>
      </el-form-item>
    </el-form>
  </div>
</template>
