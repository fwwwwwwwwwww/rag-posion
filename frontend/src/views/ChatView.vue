<template>
  <el-row :gutter="16" class="chat-view">
    <!-- 左侧：提问与 Agent 配置 -->
    <el-col :span="7">
      <el-card header="Agent 配置">
        <el-form label-width="90px" label-position="left">
          <el-form-item label="攻击场景">
            <el-select v-model="form.attack_index" style="width: 100%">
              <el-option
                v-for="a in attacks" :key="a.index" :value="a.index"
                :label="`[${a.domain}] ${a.question}`" />
            </el-select>
          </el-form-item>
          <el-form-item label="问题">
            <el-input v-model="form.question" type="textarea" :rows="3" />
          </el-form-item>
          <el-form-item label="工具上限">
            <el-slider v-model="form.max_steps" :min="1" :max="12" show-input :show-input-controls="false" />
          </el-form-item>
          <el-form-item label="会话记忆">
            <el-switch v-model="useSession" />
            <span class="hint">开启后可多轮追问</span>
          </el-form-item>
          <el-button type="primary" :loading="running" @click="ask" style="width: 100%">
            {{ running ? 'Agent 推理中…' : '发送' }}
          </el-button>
          <el-button v-if="useSession" text size="small" style="width: 100%; margin: 8px 0 0"
                     @click="resetSession">
            清空会话记忆
          </el-button>
        </el-form>

        <el-divider />
        <div class="agent-meta">
          <div class="meta-row">
            <span class="k">模型</span>
            <span class="v">{{ status.model || '-' }}</span>
          </div>
          <div class="meta-row">
            <span class="k">工具通道</span>
            <el-tag size="small" :type="status.protocol === 'native' ? 'success' : 'warning'">
              {{ status.protocol === 'native' ? '原生 tool_calls' : (status.protocol || '未就绪') }}
            </el-tag>
          </div>
          <div class="meta-row">
            <span class="k">可用工具</span>
            <span class="v">{{ toolNames.join('、') || '-' }}</span>
          </div>
          <div class="meta-row" v-if="sessionId">
            <span class="k">会话</span>
            <span class="v mono">{{ sessionId }}</span>
          </div>
        </div>
      </el-card>
    </el-col>

    <!-- 中间：决策轨迹（Agent 感的来源） -->
    <el-col :span="8">
      <el-card class="trace-card">
        <template #header>
          <div class="card-head">
            <span>决策轨迹（模型自主选择）</span>
            <el-tag v-if="toolSteps.length" size="small" type="info">
              {{ toolSteps.length }} 次工具调用
            </el-tag>
          </div>
        </template>
        <el-empty v-if="!steps.length" description="发送问题后，这里会实时显示 Agent 调用的工具" />
        <div v-for="(s, i) in steps" :key="i" class="step" :class="s.type">
          <div class="step-head">
            <el-icon v-if="s.type === 'tool_call'"><Right /></el-icon>
            <el-icon v-else-if="s.type === 'tool_result'"><Select /></el-icon>
            <el-icon v-else><Flag /></el-icon>
            <span class="step-tag">{{ typeLabel(s) }}</span>
            <span v-if="s.tool" class="tool-name">{{ s.tool }}</span>
          </div>
          <div v-if="s.type === 'tool_call'" class="step-body mono">
            {{ compactArgs(s.arguments) }}
          </div>
          <div v-else class="step-body">{{ s.summary }}</div>
        </div>
      </el-card>
    </el-col>

    <!-- 右侧：回答与来源 -->
    <el-col :span="9">
      <el-card header="模型回答" class="answer-card">
        <el-alert v-if="refused" type="warning" :closable="false" show-icon
                  title="Agent 判定证据不足，主动拒答" class="mb8" />
        <div class="answer-text" :class="{ refused }">{{ answer || '（等待提问）' }}</div>
      </el-card>
      <el-card header="检索依据（红色 = 投毒嫌疑）" class="sources-card">
        <el-empty v-if="!sources.length" description="暂无检索结果" />
        <div
          v-for="s in sources" :key="s.rank"
          class="source-item" :class="{ poison: s.poison_suspect }">
          <span class="rank">#{{ s.rank }}</span>
          <span>{{ s.content }}</span>
          <el-tag v-if="s.poison_suspect" type="danger" size="small">投毒嫌疑</el-tag>
        </div>
      </el-card>
    </el-col>
  </el-row>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { Flag, Right, Select } from '@element-plus/icons-vue'
import { api, streamAgentChat } from '../api'

const attacks = ref([])
const running = ref(false)
const answer = ref('')
const sources = ref([])
const steps = ref([])
const refused = ref(false)
const sessionId = ref('')
const useSession = ref(true)
const status = ref({})
const toolNames = ref([])
const form = reactive({ question: '', max_steps: 6, attack_index: 0 })

const toolSteps = computed(() => steps.value.filter((s) => s.type === 'tool_call'))

async function loadAttacks() {
  attacks.value = await api.attacks()
  if (attacks.value.length) {
    form.attack_index = attacks.value[0].index
    form.question = attacks.value[0].question
  }
}

async function loadAgentMeta() {
  try {
    status.value = await api.agentStatus()
    const t = await api.agentTools()
    toolNames.value = (t.tools || []).map((x) => x.name)
  } catch (e) {
    /* Agent 未就绪时静默，避免打扰 */
  }
}

function onAttackChange() {
  const a = attacks.value.find((x) => x.index === form.attack_index)
  if (a) form.question = a.question
}

watch(() => form.attack_index, onAttackChange)

function typeLabel(s) {
  if (s.type === 'tool_call') return '调用工具'
  if (s.type === 'tool_result') return '工具返回'
  if (s.type === 'finalize') return '收束生成'
  return s.type
}

function compactArgs(args) {
  if (!args || typeof args !== 'object') return ''
  return Object.entries(args)
    .map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`)
    .join('  ')
}

async function ask() {
  if (!form.question.trim() || running.value) return
  running.value = true
  answer.value = ''
  sources.value = []
  steps.value = []
  refused.value = false

  const payload = { question: form.question, max_steps: form.max_steps }
  if (useSession.value && sessionId.value) payload.session_id = sessionId.value

  await streamAgentChat(payload, {
    onSession: (id) => { sessionId.value = id },
    onStep: (s) => { steps.value = [...steps.value, s] },
    onSources: (s) => { sources.value = s },
    onDone: (d) => {
      running.value = false
      answer.value = d.answer
      refused.value = !!d.refused
      if (d.session_id) sessionId.value = d.session_id
      loadAgentMeta()
    },
    onError: (msg) => {
      running.value = false
      ElMessage.error(msg)
    },
  })
}

async function resetSession() {
  if (!sessionId.value) return
  try {
    await api.resetAgentSession(sessionId.value)
    ElMessage.success('会话记忆已清空')
    sessionId.value = ''
  } catch (e) {
    ElMessage.error(e.message)
  }
}

onMounted(() => {
  loadAttacks()
  loadAgentMeta()
})
</script>

<style scoped>
.chat-view .el-card { margin-bottom: 16px; }
.answer-card { min-height: 240px; }
.answer-text { white-space: pre-wrap; line-height: 1.8; min-height: 180px; color: #1e293b; }
.answer-text.refused { color: #b45309; }
.trace-card { min-height: 420px; }
.card-head { display: flex; justify-content: space-between; align-items: center; }
.step {
  padding: 8px 10px; border-radius: 8px; margin-bottom: 8px;
  background: #f8fafc; border: 1px solid #e2e8f0; font-size: 13px;
}
.step.tool_call { background: #eef2ff; border-color: #c7d2fe; }
.step.tool_result { background: #f0fdf4; border-color: #bbf7d0; }
.step.finalize { background: #fffbeb; border-color: #fde68a; }
.step-head { display: flex; align-items: center; gap: 6px; color: #334155; }
.step-tag { font-weight: 600; }
.tool-name { font-family: Consolas, Monaco, monospace; color: #4338ca; }
.step-body { margin-top: 4px; color: #475569; line-height: 1.6; word-break: break-all; }
.mono { font-family: Consolas, Monaco, monospace; font-size: 12px; }
.source-item {
  padding: 8px 10px; border-radius: 8px; margin-bottom: 8px;
  background: #f8fafc; border: 1px solid #e2e8f0; font-size: 13px;
  display: flex; gap: 8px; align-items: flex-start; line-height: 1.6;
}
.source-item.poison { background: #fef2f2; border-color: #fecaca; }
.rank { color: #64748b; flex-shrink: 0; }
.agent-meta { font-size: 12px; color: #475569; }
.meta-row { display: flex; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
.meta-row .k { color: #94a3b8; }
.meta-row .v { text-align: right; word-break: break-all; }
.hint { margin-left: 8px; color: #94a3b8; font-size: 12px; }
.mb8 { margin-bottom: 8px; }
</style>
