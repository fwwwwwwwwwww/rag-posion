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

    <!-- 对比图对话框 -->
    <el-dialog v-model="compareVisible" title="实验指标对比" width="720px">
      <div ref="chartRef" class="compare-chart" />
    </el-dialog>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'
import { api } from '../api'

const form = reactive({ name: '', exp_type: 'attack', model_name: 'deepseek-r1:latest' })
const autoStart = ref(true)
const experiments = ref([])
const selected = ref([])
const drawerVisible = ref(false)
const compareVisible = ref(false)
const detail = ref({})
const results = ref([])
const chartRef = ref(null)
let pollTimer = null

function statusType(status) {
  return { pending: 'info', running: 'warning', completed: 'success', failed: 'danger' }[status] || 'info'
}

async function loadList() {
  experiments.value = await api.listExperiments({ limit: 50 })
}

async function create() {
  if (!form.name.trim()) return ElMessage.warning('请输入实验名称')
  await api.createExperiment(form, autoStart.value)
  ElMessage.success('实验已创建')
  loadList()
}

async function start(row) {
  await api.startExperiment(row.id)
  ElMessage.success(`实验 #${row.id} 已开始`)
  loadList()
  startPolling()
}

function startPolling() {
  clearInterval(pollTimer)
  pollTimer = setInterval(async () => {
    await loadList()
    if (!experiments.value.some((e) => e.status === 'running')) clearInterval(pollTimer)
  }, 5000)
}

async function pollOnce(row) {
  experiments.value = await api.listExperiments({ limit: 50 })
}

async function showDetail(row) {
  detail.value = row
  results.value = await api.getExperimentResults(row.id)
  drawerVisible.value = true
}

function onSelectionChange(rows) {
  selected.value = rows
}

async function compare() {
  compareVisible.value = true
  const data = await api.compareExperiments(selected.value.map((e) => e.id))
  // 等待 DOM 渲染后画图
  setTimeout(() => {
    const chart = echarts.init(chartRef.value)
    const metricKeys = [...new Set(data.flatMap((d) => Object.keys(d.avg_metrics)))]
    chart.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: data.map((d) => `#${d.id} ${d.name}`) },
      grid: { left: 60, right: 20, bottom: 80 },
      xAxis: { type: 'category', data: metricKeys, axisLabel: { rotate: 30 } },
      yAxis: { type: 'value' },
      series: data.map((d) => ({
        name: `#${d.id} ${d.name}`,
        type: 'bar',
        data: metricKeys.map((k) => d.avg_metrics[k] ?? 0),
      })),
    })
  }, 100)
}

onMounted(loadList)
</script>

<style scoped>
.mt12 { margin-top: 12px; }
.mb12 { margin-bottom: 12px; }
.metric-tag { margin: 2px 4px 2px 0; }
.compare-chart { width: 100%; height: 420px; }
</style>
