<template>
  <el-row :gutter="16" class="chat-view">
    <!-- 左侧：提问与攻击配置 -->
    <el-col :span="8">
      <el-card header="攻击配置">
        <el-form label-width="90px" label-position="left">
          <el-form-item label="攻击场景">
            <el-select v-model="form.attack_index" style="width: 100%">
              <el-option
                v-for="a in attacks" :key="a.index" :value="a.index"
                :label="`[${a.domain}] ${a.question}`" />
            </el-select>
          </el-form-item>
          <el-form-item label="注入投毒">
            <el-switch v-model="form.poisoned" />
            <el-tag v-if="form.poisoned" type="danger" size="small" class="ml8">投毒模式</el-tag>
          </el-form-item>
          <el-form-item label="检索 K">
            <el-slider v-model="form.k" :min="1" :max="20" show-input :show-input-controls="false" />
          </el-form-item>
          <el-form-item label="问题">
            <el-input v-model="form.question" type="textarea" :rows="3" />
          </el-form-item>
          <el-button type="primary" :loading="streaming" @click="ask" style="width: 100%">
            {{ streaming ? '回答中…' : '发送' }}
          </el-button>
        </el-form>
      </el-card>
    </el-col>

    <!-- 右侧：回答与检索来源 -->
    <el-col :span="16">
      <el-card header="模型回答" class="answer-card">
        <div class="answer-text">{{ answer || '（等待提问）' }}</div>
      </el-card>
      <el-card header="检索来源（红色 = 命中投毒文档）" class="sources-card">
        <el-empty v-if="!sources.length" description="暂无检索结果" />
        <div
          v-for="s in sources" :key="s.rank"
          class="source-item" :class="{ poison: s.is_poison }">
          <span class="rank">#{{ s.rank }}</span>
          <span>{{ s.content }}</span>
          <el-tag v-if="s.is_poison" type="danger" size="small">投毒</el-tag>
        </div>
      </el-card>
    </el-col>
  </el-row>
</template>

<script setup>
import { onMounted, reactive, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { api, streamChat } from '../api'

const attacks = ref([])
const streaming = ref(false)
const answer = ref('')
const sources = ref([])
const form = reactive({ question: '', k: 3, poisoned: false, attack_index: 0 })

async function loadAttacks() {
  attacks.value = await api.attacks()
  if (attacks.value.length) {
    form.attack_index = attacks.value[0].index
    form.question = attacks.value[0].question
  }
}

function onAttackChange() {
  const a = attacks.value.find((a) => a.index === form.attack_index)
  if (a) form.question = a.question
}

watch(() => form.attack_index, onAttackChange)

async function ask() {
  if (!form.question.trim() || streaming.value) return
  streaming.value = true
  answer.value = ''
  sources.value = []
  await streamChat(form, {
    onSources: (s) => (sources.value = s),
    onToken: (t) => (answer.value += t),
    onDone: () => (streaming.value = false),
    onError: (msg) => {
      streaming.value = false
      ElMessage.error(msg)
    },
  })
}

onMounted(loadAttacks)
</script>

<style scoped>
.chat-view .el-card { margin-bottom: 16px; }
.answer-card { min-height: 260px; }
.answer-text { white-space: pre-wrap; line-height: 1.8; min-height: 200px; color: #1e293b; }
.source-item {
  padding: 8px 10px; border-radius: 8px; margin-bottom: 8px;
  background: #f8fafc; border: 1px solid #e2e8f0; font-size: 13px;
  display: flex; gap: 8px; align-items: flex-start; line-height: 1.6;
}
.source-item.poison { background: #fef2f2; border-color: #fecaca; }
.rank { color: #64748b; flex-shrink: 0; }
.ml8 { margin-left: 8px; }
</style>
