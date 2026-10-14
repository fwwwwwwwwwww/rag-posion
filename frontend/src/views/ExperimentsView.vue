<template>
  <div>
    <el-row :gutter="16">
      <!-- 创建实验 -->
      <el-col :span="8">
        <el-card header="创建实验">
          <el-form label-width="80px" label-position="left">
            <el-form-item label="名称">
              <el-input v-model="form.name" placeholder="实验名称" />
            </el-form-item>
            <el-form-item label="类型">
              <el-select v-model="form.exp_type" style="width: 100%">
                <el-option label="Baseline 对照（无毒）" value="baseline" />
                <el-option label="Blackbox 攻击" value="attack" />
                <el-option label="防御测试（k/chunk）" value="defense" />
                <el-option label="检测模型评估" value="detection" />
              </el-select>
            </el-form-item>
            <el-form-item label="模型">
              <el-input v-model="form.model_name" />
            </el-form-item>
            <el-checkbox v-model="autoStart" class="mb12">创建后立即后台执行</el-checkbox>
            <el-button type="primary" @click="create" style="width: 100%">创建实验</el-button>
          </el-form>
        </el-card>
      </el-col>

      <!-- 实验列表 -->
      <el-col :span="16">
        <el-card header="实验列表">
          <el-table :data="experiments" size="small" @selection-change="onSelectionChange">
            <el-table-column type="selection" width="42" />
            <el-table-column prop="id" label="ID" width="50" />
            <el-table-column prop="name" label="名称" min-width="140" show-overflow-tooltip />
            <el-table-column prop="exp_type" label="类型" width="90" />
            <el-table-column prop="status" label="状态" width="90">
              <template #default="{ row }">
                <el-tag :type="statusType(row.status)" size="small">{{ row.status }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="model_name" label="模型" width="150" show-overflow-tooltip />
            <el-table-column label="操作" width="220">
              <template #default="{ row }">
                <el-button size="small" :disabled="row.status === 'running'" @click="start(row)">执行</el-button>
                <el-button size="small" @click="showDetail(row)">详情</el-button>
                <el-button size="small" :disabled="row.status !== 'completed'" @click="pollOnce(row)">刷新</el-button>
              </template>
            </el-table-column>
          </el-table>
          <el-button
            type="warning" class="mt12" :disabled="selected.length < 2"
            @click="compare">
            对比选中实验（{{ selected.length }}）
          </el-button>
        </el-card>
      </el-col>
    </el-row>

    <!-- 详情抽屉 -->
    <el-drawer v-model="drawerVisible" :title="`实验 #${detail.id ?? ''} 结果`" size="55%">
      <el-table :data="results" size="small" max-height="600">
        <el-table-column prop="domain" label="领域" width="100" />
        <el-table-column prop="question" label="问题" min-width="200" show-overflow-tooltip />
        <el-table-column label="指标" min-width="260">
          <template #default="{ row }">
            <el-tag
              v-for="(v, k) in row.metrics" :key="k" size="small" class="metric-tag"
              :type="typeof v === 'number' && v > 0 ? 'danger' : 'info'">
              {{ k }}: {{ typeof v === 'number' ? v.toFixed(3) : v }}
            </el-tag>
          </template>
        </el-table-column>
      </el-table>
    </el-drawer>

    <!-- 对比图对话框（用轻量 CSS 柱状图，不再引入 echarts） -->
    <el-dialog v-model="compareVisible" title="实验指标对比" width="760px">
      <el-empty v-if="!compareData.length" description="请先选择两个以上实验进行对比" />
      <template v-else>
        <MetricBars :metrics="compareMetrics" :rows="compareRows" x-field="label"
                    :height="150" />
        <el-divider />
        <el-table :data="compareData" size="small">
          <el-table-column prop="id" label="ID" width="60" />
          <el-table-column prop="name" label="实验" min-width="140" show-overflow-tooltip />
          <el-table-column prop="exp_type" label="类型" width="90" />
          <el-table-column prop="result_count" label="结果数" width="80" />
          <el-table-column label="平均指标" min-width="240">
            <template #default="{ row }">
              <el-tag v-for="(v, k) in row.avg_metrics" :key="k" size="small"
                      class="metric-tag">
                {{ k }}: {{ Number(v).toFixed(3) }}
              </el-tag>
              <span v-if="!Object.keys(row.avg_metrics || {}).length" class="muted">-</span>
            </template>
          </el-table-column>
        </el-table>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api'
import MetricBars from '../components/MetricBars.vue'

const form = reactive({ name: '', exp_type: 'attack', model_name: 'qwen3.5:2b' })
const autoStart = ref(true)
const experiments = ref([])
const selected = ref([])
const drawerVisible = ref(false)
const compareVisible = ref(false)
const detail = ref({})
const results = ref([])
const compareData = ref([])
let pollTimer = null

// 对比图用的指标定义（每个指标一组柱，x 轴是实验）
const compareMetrics = computed(() => {
  const keys = [...new Set(compareData.value.flatMap(
    (d) => Object.keys(d.avg_metrics || {})))]
  const colors = ['blue', 'green', 'red', 'amber', 'violet']
  return keys.map((k, i) => ({
    key: k, label: k, cls: colors[i % colors.length],
  }))
})

const compareRows = computed(() => compareData.value.map((d) => ({
  label: `#${d.id}`,
  ...(d.avg_metrics || {}),
})))

function statusType(status) {
  return { pending: 'info', running: 'warning', completed: 'success', failed: 'danger' }[status] || 'info'
}

async function loadList() {
  try {
    experiments.value = await api.listExperiments({ limit: 50 })
  } catch (e) {
    ElMessage.error(e.message)
  }
}

async function create() {
  if (!form.name.trim()) return ElMessage.warning('请输入实验名称')
  try {
    await api.createExperiment(form, autoStart.value)
    ElMessage.success('实验已创建')
    if (autoStart.value) startPolling()
    await loadList()
  } catch (e) {
    ElMessage.error(e.message)
  }
}

async function start(row) {
  try {
    await api.startExperiment(row.id)
    ElMessage.success(`实验 #${row.id} 已开始`)
    await loadList()
    startPolling()
  } catch (e) {
    ElMessage.error(e.message)
  }
}

function startPolling() {
  clearInterval(pollTimer)
  pollTimer = setInterval(async () => {
    await loadList()
    // 没有 running 的实验就停止轮询，避免无意义的请求
    if (!experiments.value.some((e) => e.status === 'running')) {
      clearInterval(pollTimer)
      pollTimer = null
    }
  }, 5000)
}

async function pollOnce() {
  await loadList()
}

async function showDetail(row) {
  detail.value = row
  try {
    results.value = await api.getExperimentResults(row.id)
  } catch (e) {
    results.value = []
    ElMessage.error(e.message)
  }
  drawerVisible.value = true
}

function onSelectionChange(rows) {
  selected.value = rows
}

async function compare() {
  if (selected.value.length < 2) {
    return ElMessage.warning('请至少选择两个实验')
  }
  try {
    compareData.value = await api.compareExperiments(selected.value.map((e) => e.id))
    compareVisible.value = true
  } catch (e) {
    ElMessage.error(e.message)
  }
}

onMounted(loadList)
// 离开页面必须清掉定时器，否则会一直轮询
onBeforeUnmount(() => clearInterval(pollTimer))
</script>

<style scoped>
.metric-tag { margin: 2px 4px 2px 0; }
</style>
