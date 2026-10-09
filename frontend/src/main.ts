import { createApp } from 'vue'
import { createPinia } from 'pinia'
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate'
import ElementPlus from 'element-plus'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

import 'element-plus/dist/index.css'
import 'highlight.js/styles/github.css'
import '@/styles/main.css'

import App from './App.vue'
import { router } from './router'

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
app.mount('#app')
