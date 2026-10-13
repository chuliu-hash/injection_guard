<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'

import AppLayout from '@/layouts/AppLayout.vue'
import { useAuthStore } from '@/stores/auth'

const route = useRoute()
const auth = useAuthStore()
/** 公开页（登录）不套应用外壳，避免未登录时加载会话列表。 */
const isPublic = computed(() => route.meta.public === true)
/**
 * 登录态恢复（refresh Cookie 换 access token）完成前不渲染应用外壳：
 * 外壳里的会话列表 / 聊天页会立刻发带鉴权的请求，早于令牌就绪就是 401。
 */
const ready = computed(() => auth.ready)
</script>

<template>
  <router-view v-if="isPublic" />
  <AppLayout v-else-if="ready" />
</template>
