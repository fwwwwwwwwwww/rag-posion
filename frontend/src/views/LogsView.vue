<template>
  <el-card header="攻击日志（问答流水）">
    <el-button size="small" class="mb12" @click="load">刷新</el-button>
    <el-table :data="logs" size="small" max-height="640">
      <el-table-column prop="id" label="ID" width="60" />
      <el-table-column prop="created_at" label="时间" width="170">
        <template #default="{ row }">{{ (row.created_at || '').replace('T', ' ').slice(0, 19) }}</template>
      </el-table-column>
      <el-table-column prop="is_poison" label="投毒" width="80">
        <template #default="{ row }">
          <el-tag :type="row.is_poison ? 'danger' : 'success'" size="small">
            {{ row.is_poison ? '投毒' : '正常' }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="question" label="问题" min-width="180" show-overflow-tooltip />
      <el-table-column prop="answer" label="回答" min-width="240" show-overflow-tooltip />
      <el-table-column type="expand" label="来源" width="70">
        <template #default="{ row }">
          <div
            v-for="s in row.sources || []" :key="s.rank"
            class="src-item" :class="{ poison: s.is_poison }">
            <b>#{{ s.rank }}</b> {{ s.content }}
          </div>
        </template>
      </el-table-column>
    </el-table>
  </el-card>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { api } from '../api'

const logs = ref([])

async function load() {
  logs.value = await api.logs({ limit: 100 })
}

onMounted(load)
</script>

<style scoped>
.mb12 { margin-bottom: 12px; }
.src-item {
  padding: 6px 10px; margin: 4px 0; font-size: 12px; line-height: 1.6;
  background: #f8fafc; border-radius: 6px; border: 1px solid #e2e8f0;
}
.src-item.poison { background: #fef2f2; border-color: #fecaca; }
</style>
