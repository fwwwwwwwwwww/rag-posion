<template>
  <div class="login-page">
    <!-- 左侧品牌区（大屏显示） -->
    <div class="brand-panel">
      <div class="brand-inner">
        <div class="brand-logo">🛡️</div>
        <h1>医疗 RAG 投毒攻防平台</h1>
        <p class="brand-sub">
          工具化 Agent 防御 · 自适应投毒攻击 · 分层对抗评估
        </p>
        <ul class="feature-list">
          <li><el-icon><Search /></el-icon>自主检索与证据收集</li>
          <li><el-icon><WarningFilled /></el-icon>投毒内容检测与交叉验证</li>
          <li><el-icon><Aim /></el-icon>自适应攻击策略迭代</li>
          <li><el-icon><CircleCheck /></el-icon>四层分层对抗指标</li>
        </ul>
      </div>
      <div class="brand-foot">仅限授权研究用途</div>
    </div>

    <!-- 右侧表单 -->
    <div class="form-panel">
      <div class="form-box">
        <div class="form-head">
          <h2>登录</h2>
          <p class="muted">请使用平台账号登录</p>
        </div>

        <el-alert v-if="needsSetup" type="warning" :closable="false" show-icon
                  class="mb12">
          <template #title>系统尚未创建任何账号</template>
          <div class="hint">
            首次登录请查看<b>服务端启动日志</b>中打印的初始管理员账号与随机密码，
            登录后请立即在「账号设置」中修改密码。
          </div>
        </el-alert>

        <el-form ref="formRef" :model="form" :rules="rules" size="large"
                 @submit.prevent="onSubmit">
          <el-form-item prop="username">
            <el-input v-model="form.username" placeholder="用户名" clearable
                      :prefix-icon="User" autocomplete="username" />
          </el-form-item>

          <el-form-item prop="password">
            <el-input v-model="form.password" type="password" placeholder="密码"
                      show-password :prefix-icon="Lock"
                      autocomplete="current-password"
                      @keyup.enter="onSubmit" />
          </el-form-item>

          <el-button type="primary" size="large" class="submit-btn"
                     :loading="loading" @click="onSubmit">
            {{ loading ? '登录中…' : '登 录' }}
          </el-button>
        </el-form>

        <div class="form-foot">
          <span class="hint">
            <el-icon><Lock /></el-icon>
            密码经 bcrypt 加盐哈希存储，平台不保存明文
          </span>
        </div>

        <el-alert v-if="errorMsg" type="error" :closable="false" show-icon
                  class="mt12" :title="errorMsg" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Lock, User } from '@element-plus/icons-vue'
import { auth } from '../api'
import { login } from '../stores/auth'

const route = useRoute()
const router = useRouter()

const formRef = ref()
const loading = ref(false)
const errorMsg = ref('')
const needsSetup = ref(false)

const form = reactive({ username: '', password: '' })

const rules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }],
}

async function onSubmit() {
  errorMsg.value = ''
  try {
    await formRef.value.validate()
  } catch {
    return
  }

  loading.value = true
  try {
    const user = await login(form.username.trim(), form.password)
    ElMessage.success(`欢迎回来，${user.display_name || user.username}`)
    // 登录后回跳到原目标页
    const target = route.query.redirect
    await router.replace(typeof target === 'string' && target.startsWith('/')
      ? target : '/chat')
  } catch (e) {
    // 后端统一返回"用户名或密码不正确"，不泄露账号是否存在
    errorMsg.value = e.message || '登录失败'
    form.password = ''
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  try {
    const r = await auth.needsSetup?.()
    needsSetup.value = !!r?.needs_setup
  } catch {
    /* 忽略 */
  }
})
</script>

<style scoped>
.login-page {
  display: grid;
  grid-template-columns: 1.1fr .9fr;
  height: 100vh;
}

/* ---------- 左侧品牌 ---------- */
.brand-panel {
  position: relative;
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 0 64px;
  color: #fff;
  background:
    radial-gradient(1200px 600px at 10% 0%, #6366f1 0%, transparent 55%),
    linear-gradient(150deg, #312e81 0%, #4f46e5 55%, #7c3aed 100%);
  overflow: hidden;
}
.brand-panel::after {
  /* 装饰性光斑 */
  content: '';
  position: absolute;
  right: -120px; bottom: -160px;
  width: 420px; height: 420px;
  border-radius: 50%;
  background: rgba(255, 255, 255, .08);
}
.brand-inner { position: relative; z-index: 1; max-width: 460px; }
.brand-logo {
  width: 56px; height: 56px;
  display: grid; place-items: center;
  font-size: 27px;
  border-radius: 15px;
  background: rgba(255, 255, 255, .16);
  backdrop-filter: blur(6px);
  margin-bottom: 22px;
}
.brand-inner h1 { font-size: 27px; font-weight: 700; letter-spacing: .5px; }
.brand-sub { margin-top: 10px; font-size: 14px; opacity: .85; line-height: 1.7; }
.feature-list { list-style: none; margin-top: 34px; }
.feature-list li {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 13px; font-size: 14px; opacity: .92;
}
.feature-list .el-icon { font-size: 16px; opacity: .9; }
.brand-foot {
  position: absolute; left: 64px; bottom: 28px;
  font-size: 12px; opacity: .6;
}

/* ---------- 右侧表单 ---------- */
.form-panel {
  display: grid;
  place-items: center;
  padding: 24px;
  background: var(--bg-card);
}
.form-box { width: 100%; max-width: 360px; }
.form-head { margin-bottom: 26px; }
.form-head h2 { font-size: 22px; font-weight: 650; margin-bottom: 6px; }
.submit-btn { width: 100%; margin-top: 4px; letter-spacing: 2px; }
.form-foot {
  margin-top: 18px;
  padding-top: 14px;
  border-top: 1px dashed var(--border);
  display: flex; justify-content: center;
}
.form-foot .hint { display: flex; align-items: center; gap: 5px; }

/* ---------- 响应式：窄屏隐藏品牌区 ---------- */
@media (max-width: 860px) {
  .login-page { grid-template-columns: 1fr; }
  .brand-panel { display: none; }
}
</style>
