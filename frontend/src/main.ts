import { createApp } from 'vue'
import { createPinia } from 'pinia'
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate'
import ElementPlus from 'element-plus'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

import 'element-plus/dist/index.css'
import 'highlight.js/styles/github.css'
import '@/styles/main.css'

import App from './App.vue'
import { registerRefreshHandler } from './api/authToken'
import { router } from './router'
import { useAuthStore } from './stores/auth'

const app = createApp(App)

// 全局注册 Element Plus 图标（模板里可直接用 <el-icon><Shield /></el-icon> 等）
for (const [name, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(name, component)
}

const pinia = createPinia()
pinia.use(piniaPluginPersistedstate)

app.use(pinia)
app.use(router)
app.use(ElementPlus)

// http 层 401 时用 refresh Cookie 换新令牌（注册在 pinia 就绪之后）
registerRefreshHandler(() => useAuthStore().refreshSession())

// 等首次导航完成再挂载：路由守卫会在其中用 refresh Cookie 恢复登录态（access token 只在内存），
// 否则应用外壳会在拿到令牌前挂载，首屏的带鉴权请求必然 401（再靠 401 重试补救）。
void router.isReady().finally(() => app.mount('#app'))
