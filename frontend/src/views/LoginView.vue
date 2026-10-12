<script setup lang="ts">
import { reactive, ref } from 'vue'
import { Lock, User as UserIcon } from '@element-plus/icons-vue'
import { useRoute, useRouter } from 'vue-router'

import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
const route = useRoute()
const router = useRouter()

const mode = ref<'login' | 'register'>('login')
const form = reactive({ username: '', password: '' })
const error = ref('')

function toggleMode(): void {
  mode.value = mode.value === 'login' ? 'register' : 'login'
  error.value = ''
}

async function submit(): Promise<void> {
  error.value = ''
  if (!form.username.trim() || !form.password) {
    error.value = '请填写用户名与密码'
    return
  }
  try {
    if (mode.value === 'login') await auth.login(form.username.trim(), form.password)
    else await auth.register(form.username.trim(), form.password)
    const redirect = typeof route.query.redirect === 'string' ? route.query.redirect : '/chat'
    await router.replace(redirect)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}
</script>

<template>
  <div class="login">
    <el-card class="login__card">
      <h2 class="login__title">InjectionGuard · LLM 安全护栏</h2>
      <p class="login__subtitle">{{ mode === 'login' ? '登录以继续' : '注册新账号' }}</p>

      <el-form label-position="top" @submit.prevent="submit">
        <el-form-item label="用户名">
          <el-input v-model="form.username" :prefix-icon="UserIcon" autocomplete="username" />
        </el-form-item>
        <el-form-item label="密码">
          <el-input
            v-model="form.password"
            type="password"
            show-password
            :prefix-icon="Lock"
            :autocomplete="mode === 'login' ? 'current-password' : 'new-password'"
            @keyup.enter="submit"
          />
        </el-form-item>
      </el-form>

      <el-alert v-if="error" :title="error" type="error" :closable="false" show-icon />

      <el-button type="primary" class="login__submit" :loading="auth.loading" @click="submit">
        {{ mode === 'login' ? '登录' : '注册并登录' }}
      </el-button>
      <div class="login__switch">
        <el-button link type="primary" @click="toggleMode">
          {{ mode === 'login' ? '没有账号？去注册' : '已有账号？去登录' }}
        </el-button>
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.login {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
}
.login__card {
  width: 360px;
}
.login__title {
  margin: 0 0 4px;
  text-align: center;
}
.login__subtitle {
  margin: 0 0 20px;
  text-align: center;
  color: var(--el-text-color-secondary);
}
.login__submit {
  width: 100%;
}
.login__switch {
  margin-top: 8px;
  text-align: center;
}
</style>
