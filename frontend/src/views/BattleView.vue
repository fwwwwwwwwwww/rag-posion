<template>
  <div>
    <!-- 顶部：启动战役 + 防御变体说明 -->
    <el-row :gutter="16">
      <el-col :span="9">
        <el-card header="启动自适应攻击战役">
          <el-form label-width="96px" label-position="left">
            <el-form-item label="攻击场景">
              <el-select v-model="form.attack_index" style="width: 100%">
                <el-option
                  v-for="a in attacks" :key="a.index" :value="a.index"
                  :label="`[${a.domain}] ${a.question}`" />
              </el-select>
            </el-form-item>
            <el-form-item label="目标防御">
              <el-select v-model="form.defense_variant" style="width: 100%">
                <el-option
                  v-for="v in variants" :key="v.name" :value="v.name"
                  :label="`${v.name} — ${v.note}`" />
              </el-select>
            </el-form-item>
            <el-form-item label="最大轮数">
              <el-slider v-model="form.max_rounds" :min="1" :max="10" show-input
                         :show-input-controls="false" />
            </el-form-item>
            <el-form-item label="投放篇数">
              <el-slider v-model="form.doc_count" :min="5" :max="50" show-input
                         :show-input-controls="false" />
            </el-form-item>
            <el-button type="danger" :loading="starting" @click="startCampaign"
                       style="width: 100%">
              {{ starting ? '已提交…' : '启动攻击' }}
            </el-button>
            <el-button text size="small" style="width: 100%; margin: 8px 0 0"
                       @click="loadIterations">
              刷新轨迹
            </el-button>
          </el-form>
        </el-card>
      </el-col>

      <el-col :span="15">
        <el-card header="攻击策略（模型会按失败原因自行选择）">
          <el-table :data="strategies" size="small">
            <el-table-column prop="name" label="策略" width="120" />
            <el-table-column prop="note" label="作用方向" />
          </el-table>
          <el-divider />
          <div class="hint">
            攻击 Agent 每轮读取四层指标，判断"攻击死在哪一层"，再换策略：
            <b>没进检索</b>→术语化 · <b>被检测</b>→规避改写 ·
            <b>核查不过</b>→拆多篇刷独立来源 · <b>被模型识破</b>→伪装研究设计
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 攻击轨迹 -->
    <el-card class="mt16">
      <template #header>
        <div class="card-head">
          <span>攻击迭代轨迹（每轮换了什么策略、指标如何变化）</span>
          <el-tag v-if="iterations.length" size="small" type="info">
            {{ iterations.length }} 轮
          </el-tag>
        </div>
      </template>
      <el-empty v-if="!iterations.length"
                description="还没有攻击记录，启动一次攻击战役后这里会显示轨迹" />

      <template v-else>
        <!-- 指标趋势（用 CSS 柱状，避免额外图表依赖） -->
        <div class="metrics">
          <div v-for="m in trendMetrics" :key="m.key" class="metric">
            <div class="metric-label">{{ m.label }}</div>
            <div class="bars">
              <div v-for="(row, i) in chronological" :key="i" class="bar-wrap"
                   :title="`第${row.round_index}轮: ${(row[m.key] ?? 0)}`">
                <div class="bar" :style="{ height: barHeight(row[m.key]) }"
                     :class="m.cls" />
                <span class="bar-idx">{{ row.round_index }}</span>
              </div>
            </div>
            <div class="metric-vals">
              {{ chronological.map((r) => fmt(r[m.key])).join(' · ') }}
            </div>
          </div>
        </div>

        <el-table :data="chronological" size="small" class="mt16">
          <el-table-column prop="round_index" label="轮" width="46" />
          <el-table-column prop="strategy" label="策略" width="112" />
          <el-table-column label="召回@k" width="86">
            <template #default="{ row }">{{ fmt(row.recall_at_k) }}</template>
          </el-table-column>
          <el-table-column label="检出率" width="78">
            <template #default="{ row }">
              <span :class="{ good: row.detection_rate > 0 }">
                {{ fmt(row.detection_rate) }}
              </span>
            </template>
          </el-table-column>
          <el-table-column label="核查通过" width="84">
            <template #default="{ row }">{{ fmt(row.survive) }}</template>
          </el-table-column>
          <el-table-column label="ASR" width="60">
            <template #default="{ row }">
              <el-tag :type="row.asr ? 'danger' : 'success'" size="small">
                {{ row.asr ? '成功' : '失败' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="判定" width="104">
            <template #default="{ row }">
              <el-tag size="small" :type="verdictType(row.verdict)">
                {{ row.verdict }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="failure_layer" label="死在" width="130" />
          <el-table-column prop="strategy_reason" label="换策略理由"
                           min-width="200" show-overflow-tooltip />
        </el-table>

        <el-divider />
        <div class="hint">
          判读：<b>召回@k</b> 高但 <b>检出率</b> 也高 → 攻击进了检索但被检测拦住；
          <b>召回@k</b> 低 → 攻击连检索都没进（说明查询短语没匹配上）；
          <b>ASR=成功</b> 表示回答**采纳**了虚假断言（不是仅仅提到）。
        </div>
      </template>
    </el-card>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api'

const attacks = ref([])
const strategies = ref([])
const variants = ref([])
const iterations = ref([])
const starting = ref(false)
const form = reactive({
  attack_index: 0, defense_variant: 'agent', max_rounds: 3, doc_count: 15,
})

const chronological = computed(() =>
  [...iterations.value].sort((a, b) => (a.round_index - b.round_index) || (a.id - b.id)))

const trendMetrics = [
  { key: 'recall_at_k', label: '召回@k（进检索）', cls: 'bar-blue' },
  { key: 'detection_rate', label: '检出率（防御生效）', cls: 'bar-green' },
  { key: 'asr', label: 'ASR（采纳虚假断言）', cls: 'bar-red' },
]

function fmt(v) {
  return v === null || v === undefined ? '-' : Number(v).toFixed(2)
}

function barHeight(v) {
  const n = Math.max(0, Math.min(1, Number(v) || 0))
  return `${Math.max(3, n * 100)}%`
}

function verdictType(v) {
  return { endorsed: 'danger', rejected: 'success',
           mentioned_only: 'warning', unrelated: 'info' }[v] || 'info'
}

async function loadAttacks() {
  attacks.value = await api.attacks()
  if (attacks.value.length) form.attack_index = attacks.value[0].index
}

async function loadMeta() {
  try {
    strategies.value = await api.attackStrategies()
    variants.value = await api.attackVariants()
  } catch (e) { /* 静默 */ }
}

async function loadIterations() {
  try {
    iterations.value = await api.attackIterations()
  } catch (e) {
    ElMessage.error(e.message)
  }
}

async function startCampaign() {
  starting.value = true
  try {
    const r = await api.startCampaign({ ...form })
    ElMessage.success(r.message || '已提交')
    setTimeout(loadIterations, 3000)
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    starting.value = false
  }
}

onMounted(() => {
  loadAttacks()
  loadMeta()
  loadIterations()
})
</script>

<style scoped>
.mt16 { margin-top: 16px; }
.card-head { display: flex; justify-content: space-between; align-items: center; }
.hint { font-size: 12px; color: #64748b; line-height: 1.7; }
.good { color: #16a34a; font-weight: 600; }

.metrics { display: flex; gap: 28px; flex-wrap: wrap; }
.metric { min-width: 220px; }
.metric-label { font-size: 12px; color: #475569; margin-bottom: 6px; }
.bars { display: flex; gap: 8px; align-items: flex-end; height: 76px; }
.bar-wrap { display: flex; flex-direction: column; align-items: center;
            height: 100%; justify-content: flex-end; width: 26px; }
.bar { width: 100%; border-radius: 3px 3px 0 0; }
.bar-blue { background: #60a5fa; }
.bar-green { background: #34d399; }
.bar-red { background: #f87171; }
.bar-idx { font-size: 10px; color: #94a3b8; margin-top: 2px; }
.metric-vals { font-size: 11px; color: #94a3b8; margin-top: 4px;
               font-family: Consolas, Monaco, monospace; }
</style>
