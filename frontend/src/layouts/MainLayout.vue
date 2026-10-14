<template>
  <el-container class="layout">
    <!-- 顶栏 -->
    <el-header class="header">
      <button class="icon-btn" title="收起/展开菜单" @click="collapsed = !collapsed">
        <el-icon><Fold v-if="!collapsed" /><Expand v-else /></el-icon>
      </button>

      <div class="brand" @click="$router.push('/chat')">
        <div class="logo">🛡️</div>
        <div class="brand-text">
          <div class="title">医疗 RAG 投毒攻防平台</div>
          <div class="subtitle">工具化 Agent · 自主检索 / 投毒检测 / 交叉验证 / 拒答</div>
        </div>
      </div>

      <div class="spacer" />

      <!-- 系统状态 -->
      <el-tooltip :content="healthTip" placement="bottom">
        <el-tag :type="healthType" size="small" effect="light" class="health-tag">
          <span class="dot" :class="healthType" />
          {{ healthText }}
        </el-tag>
      </el-tooltip>

      <el-tooltip content="刷新状态" placement="bottom">
        <button class="icon-btn" :disabled="healthLoading" @click="refreshHealth">
          <el-icon :class="{ spinning: healthLoading }"><Refresh /></el-icon>
        </button>
      </el-tooltip>

      <!-- 用户菜单 -->
      <el-dropdown trigger="click" @command="onUserCommand">
        <div class="user">
          <el-avatar :size="28" class="avatar">{{ avatarText }}</el-avatar>
          <span class="username">{{ user?.display_name || user?.username || '未登录' }}</span>
          <el-icon><ArrowDown /></el-icon>
        </div>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item command="settings">
              <el-icon><Setting /></el-icon>账号设置
            </el-dropdown-item>
            <el-dropdown-item command="logout" divided>
              <el-icon><SwitchButton /></el-icon>退出登录
            </el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
    </el-header>

    <el-container class="body">
      <!-- 侧边导航 -->
      <el-aside :width="collapsed ? '64px' : '210px'" class="aside">
        <el-menu :default-active="activePath" router :collapse="collapsed"
                 :collapse-transition="false">
          <el-menu-item v-for="item in navItems" :key="item.path" :index="item.path">
            <el-icon><component :is="item.icon" /></el-icon>
            <template #title>{{ item.title }}</template>
          </el-menu-item>
        </el-menu>

        <div v-if="!collapsed" class="aside-foot">
          <div class="foot-line">
            <span class="k">模型</span><span class="v ellipsis">{{ status.model || '-' }}</span>
          </div>
          <div class="foot-line">
            <span class="k">通道</span>
            <span class="v">
              <el-tag size="small" :type="status.protocol === 'native' ? 'success' : 'warning'">
                {{ status.protocol || '未就绪' }}
              </el-tag>
            </span>
          </div>
          <div class="foot-line">
            <span class="k">知识库</span><span class="v">{{ status.kb_chunks || 0 }} chunk</span>
          </div>
        </div>
      </el-aside>

      <el-main class="main">
        <router-view v-slot="{ Component }">
          <transition name="fade-slide" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { api } from '../api'
import { authState, logout } from '../stores/auth'

const route = useRoute()
const router = useRouter()

const collapsed = ref(false)
const health = ref({ status: '-' })
const healthLoading = ref(false)
const status = ref({})

const user = computed(() => authState.user)
const avatarText = computed(() => {
  const n = user.value?.display_name || user.value?.username || '?'
  return n.slice(0, 1).toUpperCase()
})

const navItems = [
  { path: '/chat', title: '问答演示', icon: 'ChatDotRound' },
  { path: '/battle', title: '攻防对抗', icon: 'Aim' },
  { path: '/experiments', title: '实验管理', icon: 'DataAnalysis' },
  { path: '/poison-docs', title: '投毒文档', icon: 'Warning' },
  { path: '/logs', title: '攻击日志', icon: 'Document' },
]

const activePath = computed(() => route.path)

const healthText = computed(() => {
  const s = health.value.status
  if (s === 'ready') return 'RAG 就绪'
  if (s === 'lazy') return '懒加载'
  if (s === '-') return '检测中'
  return '离线'
})
const healthType = computed(() => {
  const s = health.value.status
  if (s === 'ready') return 'success'
  if (s === 'lazy') return 'info'
  if (s === '-') return 'info'
  return 'danger'
})
const healthTip = computed(() => {
  const h = health.value
  const parts = [`状态：${h.status || '-'}`]
  if (h.model) parts.push(`模型：${h.model}`)
  if (h.embedding_model) parts.push(`向量模型：${h.embedding_model}`)
  if (h.kb_documents) parts.push(`知识库：${h.kb_documents} 篇`)
  return parts.join('　|　')
})

async function refreshHealth() {
  healthLoading.value = true
  try {
    health.value = await api.health()
  } catch (e) {
    health.value = { status: 'offline' }
  } finally {
    healthLoading.value = false
  }
}

async function loadAgentStatus() {
  try {
    status.value = await api.agentStatus()
  } catch (e) {
    /* 未就绪时静默 */
  }
}

async function onUserCommand(cmd) {
  if (cmd === 'settings') {
    router.push('/settings')
    return
  }
  if (cmd === 'logout') {
    try {
      await ElMessageBox.confirm('确定要退出登录吗？', '提示', {
        confirmButtonText: '退出',
        cancelButtonText: '取消',
        type: 'warning',
      })
    } catch {
      return // 用户取消
    }
    await logout()
    ElMessage.success('已退出登录')
    router.replace({ name: 'login' })
  }
}

onMounted(() => {
  refreshHealth()
  loadAgentStatus()
})
</script>

<style scoped>
.layout { height: 100vh; }

/* ---------- 顶栏 ---------- */
.header {
  display: flex;
  align-items: center;
  gap: 10px;
  height: 58px;
  padding: 0 16px;
  background: #fff;
  border-bottom: 1px solid var(--border);
  box-shadow: var(--shadow-sm);
  z-index: 10;
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  user-select: none;
}
.logo {
  width: 34px; height: 34px;
  border-radius: 9px;
  font-size: 17px;
  display: grid;
  place-items: center;
  background: linear-gradient(135deg, var(--brand), var(--brand-light));
  box-shadow: 0 2px 6px rgba(79, 70, 229, .3);
}
.title { font-size: 14px; font-weight: 650; line-height: 1.25; }
.subtitle { font-size: 11px; color: var(--text-3); line-height: 1.25; }
.spacer { flex: 1; }

.icon-btn {
  display: grid;
  place-items: center;
  width: 32px; height: 32px;
  border: none;
  border-radius: 8px;
  background: transparent;
  color: var(--text-2);
  cursor: pointer;
  transition: background .15s;
}
.icon-btn:hover:not(:disabled) { background: #f1f5f9; }
.icon-btn:disabled { opacity: .5; cursor: default; }
.spinning { animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

.health-tag { display: inline-flex; align-items: center; gap: 5px; }
.dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: currentColor; opacity: .85;
}

.user {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 4px 8px;
  border-radius: 8px;
  cursor: pointer;
  outline: none;
  transition: background .15s;
}
.user:hover { background: #f1f5f9; }
.avatar {
  background: linear-gradient(135deg, var(--brand), var(--brand-light));
  color: #fff;
  font-size: 13px;
  font-weight: 600;
}
.username { font-size: 13px; color: var(--text-1); max-width: 120px;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ---------- 侧栏 ---------- */
.body { height: calc(100vh - 58px); }
.aside {
  background: #fff;
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  transition: width .18s ease;
  overflow: hidden;
}
.aside :deep(.el-menu) { border-right: none; flex: 1; }
.aside :deep(.el-menu-item) { margin: 2px 8px; border-radius: 8px; height: 42px; }
.aside :deep(.el-menu-item.is-active) {
  background: var(--brand-soft);
  color: var(--brand);
  font-weight: 600;
}
.aside :deep(.el-menu--collapse .el-menu-item) { margin: 2px 6px; }

.aside-foot {
  border-top: 1px solid var(--border);
  padding: 10px 14px;
  font-size: 12px;
}
.foot-line { display: flex; justify-content: space-between; gap: 8px; margin-bottom: 5px; }
.foot-line .k { color: var(--text-3); flex-shrink: 0; }
.foot-line .v { color: var(--text-2); text-align: right; min-width: 0; }

/* ---------- 主区 ---------- */
.main { background: var(--bg-page); overflow-y: auto; }

@media (max-width: 900px) {
  .brand-text { display: none; }
  .username { display: none; }
}
</style>
