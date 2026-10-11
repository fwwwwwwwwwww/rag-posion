<template>
  <el-container class="layout">
    <el-header class="header">
      <div class="logo">🛡️</div>
      <div>
        <div class="title">医疗 RAG 投毒攻防平台</div>
        <div class="subtitle">LangGraph 防御链路 · Vue3 + FastAPI</div>
      </div>
      <el-tag :type="health.status === 'ready' ? 'success' : 'info'" class="health-tag">
        {{ health.status === 'ready' ? 'RAG 就绪' : '懒加载' }}
        <span v-if="health.kb_documents"> · 知识库 {{ health.kb_documents }} 篇</span>
      </el-tag>
      <div class="spacer" />
      <el-input
        v-model="tokenInput"
        placeholder="Bearer Token"
        size="small"
        class="token-input"
        show-password
        @change="onTokenChange"
      />
      <el-button size="small" @click="refreshHealth">刷新状态</el-button>
    </el-header>

    <el-container>
      <el-aside width="200px" class="aside">
        <el-menu :default-active="$route.path" router>
          <el-menu-item index="/chat">
            <el-icon><ChatDotRound /></el-icon>问答演示
          </el-menu-item>
          <el-menu-item index="/experiments">
            <el-icon><DataAnalysis /></el-icon>实验管理
          </el-menu-item>
          <el-menu-item index="/poison-docs">
            <el-icon><Warning /></el-icon>投毒文档
          </el-menu-item>
          <el-menu-item index="/logs">
            <el-icon><Document /></el-icon>攻击日志
          </el-menu-item>
        </el-menu>
      </el-aside>

      <el-main class="main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { ChatDotRound, DataAnalysis, Document, Warning } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { api, getToken, setToken } from './api'

const health = ref({ status: '-' })
const tokenInput = ref(getToken())

async function refreshHealth() {
  try {
    health.value = await api.health()
  } catch (e) {
    health.value = { status: 'offline' }
  }
}

function onTokenChange(value) {
  setToken(value.trim())
  ElMessage.success('Token 已更新')
  refreshHealth()
}

onMounted(refreshHealth)
</script>

<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, "Microsoft YaHei", "PingFang SC", sans-serif; }
.layout { height: 100vh; }
.header {
  display: flex; align-items: center; gap: 12px;
  border-bottom: 1px solid #e2e8f0; background: #fff;
}
.logo {
  width: 36px; height: 36px; border-radius: 10px; font-size: 18px;
  background: linear-gradient(135deg, #6366f1, #818cf8);
  display: flex; align-items: center; justify-content: center;
}
.title { font-size: 15px; font-weight: 600; color: #1e293b; }
.subtitle { font-size: 12px; color: #64748b; }
.spacer { flex: 1; }
.token-input { width: 200px; }
.aside { border-right: 1px solid #e2e8f0; background: #fff; }
.aside .el-menu { border-right: none; }
.main { background: #f8fafc; }
</style>
