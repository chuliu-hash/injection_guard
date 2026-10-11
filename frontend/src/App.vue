<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'

import AppLayout from '@/layouts/AppLayout.vue'

const route = useRoute()
/** 公开页（登录）不套应用外壳，避免未登录时加载会话列表。 */
const isPublic = computed(() => route.meta.public === true)
</script>

<template>
  <router-view v-if="isPublic" />
  <AppLayout v-else />
</template>
