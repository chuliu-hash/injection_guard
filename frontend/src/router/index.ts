import { createRouter, createWebHistory } from 'vue-router'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/chat' },
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
