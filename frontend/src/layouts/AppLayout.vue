<script setup lang="ts">
import { Setting } from '@element-plus/icons-vue'

import BackendStatus from '@/components/BackendStatus.vue'
import SessionList from '@/components/SessionList.vue'
</script>

<template>
  <el-container class="layout">
    <el-aside width="260px" class="layout__aside">
      <SessionList />
    </el-aside>

    <el-container>
      <el-header class="layout__header">
        <div class="layout__title">AI 安全护栏 · 聊天助手</div>
        <div class="layout__right">
          <BackendStatus />
          <router-link to="/settings">
            <el-button link type="primary" :icon="Setting">设置</el-button>
          </router-link>
        </div>
      </el-header>
      <el-main class="layout__main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>
