import { createRouter, createWebHistory } from 'vue-router'

import { useAuthStore } from '@/stores/auth'

declare module 'vue-router' {
  interface RouteMeta {
    /** 公开路由（无需登录），如登录页。 */
    public?: boolean
  }
}

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/chat' },
    {
      path: '/login',
      name: 'login',
      component: () => import('@/views/LoginView.vue'),
      meta: { public: true },
    },
    {
      path: '/chat/:sessionId?',
      name: 'chat',
      component: () => import('@/views/ChatView.vue'),
    },
    {
      path: '/settings',
      name: 'settings',
      component: () => import('@/views/SettingsView.vue'),
    },
    { path: '/:pathMatch(.*)*', redirect: '/chat' },
  ],
})

/** 全局守卫：公开路由放行；其余需登录（启动时用 refresh Cookie 尝试恢复一次会话）。 */
router.beforeEach(async (to) => {
  const auth = useAuthStore()
  const authenticated = await auth.ensureReady()
  if (to.meta.public) {
    return authenticated && to.name === 'login' ? { name: 'chat' } : true
  }
  if (authenticated) return true
  return { name: 'login', query: { redirect: to.fullPath } }
})
