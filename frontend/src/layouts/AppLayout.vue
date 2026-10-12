<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { Setting, SwitchButton } from '@element-plus/icons-vue'
import { useRouter } from 'vue-router'

import BackendStatus from '@/components/BackendStatus.vue'
import SessionList from '@/components/SessionList.vue'
import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
const router = useRouter()

async function onLogout(): Promise<void> {
  await auth.logout()
  ElMessage.success('已退出登录')
  void router.push({ name: 'login' })
}
</script>

<template>
  <el-container class="layout">
    <el-aside width="260px" class="layout__aside">
      <SessionList />
    </el-aside>

    <el-container>
      <el-header class="layout__header">
        <div class="layout__title">InjectionGuard · LLM 安全护栏</div>
        <div class="layout__right">
          <BackendStatus />
          <router-link to="/settings">
            <el-button link type="primary" :icon="Setting">设置</el-button>
          </router-link>
          <el-dropdown>
            <el-button link type="primary">{{ auth.user?.username ?? '账号' }}</el-button>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item :icon="SwitchButton" @click="onLogout">退出登录</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </el-header>
      <el-main class="layout__main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>
