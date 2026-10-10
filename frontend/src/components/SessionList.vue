<script setup lang="ts">
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import { useRoute, useRouter } from 'vue-router'

import { useSessionStore } from '@/stores/session'
import { formatTime } from '@/utils/format'

const store = useSessionStore()
const route = useRoute()
const router = useRouter()

/** 切到某会话的聊天页。侧栏在任意路由下都挂着（如 /settings），
 *  故导航必须由这里发起——ChatView 在非聊天路由下已卸载，其 currentId 监听不会触发。 */
function goChat(id: string): void {
  if (route.path !== `/chat/${id}`) void router.push(`/chat/${id}`)
}

function onCreate(): void {
  goChat(store.createSession().id)
}

function onSelect(id: string): void {
  store.selectSession(id)
  goChat(id)
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
        @click="onSelect(item.id)"
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
