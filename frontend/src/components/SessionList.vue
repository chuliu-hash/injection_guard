<script setup lang="ts">
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'

import { useSessionStore } from '@/stores/session'
import { formatTime } from '@/utils/format'

const store = useSessionStore()

function onCreate(): void {
  store.createSession()
}

async function onRename(id: string, current: string): Promise<void> {
  try {
    const result = (await ElMessageBox.prompt('请输入新的会话名称', '重命名会话', {
      inputValue: current,
      confirmButtonText: '确定',
      cancelButtonText: '取消',
    })) as { value: string }
    store.renameSession(id, result.value)
  } catch {
    // 用户取消
  }
}

async function onDelete(id: string): Promise<void> {
  try {
    await ElMessageBox.confirm('删除后该会话与检测记录不可恢复，确定删除？', '删除会话', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
    })
    store.deleteSession(id)
    ElMessage.success('已删除')
  } catch {
    // 用户取消
  }
}
</script>

<template>
  <div class="session-list">
    <div class="session-list__head">
      <el-button type="primary" :icon="Plus" style="width: 100%" @click="onCreate">
        新建会话
      </el-button>
    </div>
    <div class="session-list__items">
      <el-empty v-if="store.sorted.length === 0" description="暂无会话" :image-size="60" />
      <div
        v-for="item in store.sorted"
        :key="item.id"
        class="session-item"
        :class="{ 'session-item--active': item.id === store.currentId }"
        @click="store.selectSession(item.id)"
      >
        <div class="session-item__row">
          <span class="session-item__title">{{ item.title }}</span>
          <span class="session-item__ops">
            <el-button link size="small" @click.stop="onRename(item.id, item.title)">
              改名
            </el-button>
            <el-button link type="danger" size="small" @click.stop="onDelete(item.id)">
              删除
            </el-button>
          </span>
        </div>
        <div class="session-item__time">{{ formatTime(item.updatedAt) }}</div>
      </div>
    </div>
  </div>
</template>
