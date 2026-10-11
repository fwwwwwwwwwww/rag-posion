<template>
  <div>
    <el-card header="批量生成投毒文档">
      <el-form inline>
        <el-form-item label="攻击场景">
          <el-select v-model="attackIndex" style="width: 320px">
            <el-option
              v-for="a in attacks" :key="a.index" :value="a.index"
              :label="`[${a.domain}] ${a.question}`" />
          </el-select>
        </el-form-item>
        <el-form-item label="数量">
          <el-input-number v-model="count" :min="1" :max="100" />
        </el-form-item>
        <el-form-item>
          <el-button type="danger" @click="generate">生成并入库</el-button>
          <el-button type="warning" plain @click="deleteByAttack">按场景清空</el-button>
        </el-form-item>
      </el-form>
      <el-alert
        title="投毒文档仅存入 MySQL 的 poison_documents 表，供检测模型与实验复用；注入向量库需在问答演示或实验执行中触发。"
        type="info" :closable="false" class="mb12" />
    </el-card>

    <el-card header="投毒文档列表" class="mt12">
      <el-table :data="docs" size="small" max-height="560">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="domain" label="领域" width="100" />
        <el-table-column prop="question" label="目标问题" min-width="180" show-overflow-tooltip />
        <el-table-column prop="content" label="文档内容" min-width="320" show-overflow-tooltip />
        <el-table-column prop="detected_poison" label="检测结果" width="100">
          <template #default="{ row }">
            <el-tag v-if="row.detected_poison === true" type="success" size="small">已检出</el-tag>
            <el-tag v-else-if="row.detected_poison === false" type="info" size="small">未检出</el-tag>
            <el-tag v-else type="warning" size="small">待评估</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="80">
          <template #default="{ row }">
            <el-button size="small" type="danger" plain @click="remove(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { api } from '../api'

const attacks = ref([])
const attackIndex = ref(0)
const count = ref(15)
const docs = ref([])

async function loadAll() {
  attacks.value = await api.attacks()
  docs.value = await api.listPoison({ limit: 200 })
}

async function generate() {
  const created = await api.generatePoison({ attack_index: attackIndex.value, count: count.value })
  ElMessage.success(`已生成 ${created.length} 篇投毒文档`)
  loadAll()
}

async function remove(row) {
  await ElMessageBox.confirm(`确认删除文档 #${row.id}？`, '提示', { type: 'warning' })
  await api.deletePoison(row.id)
  loadAll()
}

async function deleteByAttack() {
  await ElMessageBox.confirm(`确认清空场景 ${attackIndex.value} 的全部投毒文档？`, '提示', { type: 'warning' })
  const res = await api.deletePoisonByAttack(attackIndex.value)
  ElMessage.success(`已删除 ${res.count} 篇`)
  loadAll()
}

onMounted(loadAll)
</script>

<style scoped>
.mt12 { margin-top: 12px; }
.mb12 { margin-bottom: 12px; }
</style>
