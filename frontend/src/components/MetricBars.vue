<template>
  <!--
    轻量柱状趋势图。
    为什么不用 echarts：项目只画简单的柱状对比，而 `import * as echarts`
    会把整个图表库（约 1MB）打进包里。实测引入后产物出现两个 1MB+ 的 chunk。
    纯 CSS 柱状图零依赖、渲染更快，也更容易和整体风格统一。
  -->
  <div class="trend">
    <div v-for="m in metrics" :key="m.key" class="trend-item">
      <div class="trend-label">
        <span>{{ m.label }}</span>
        <span v-if="m.hint" class="hint">{{ m.hint }}</span>
      </div>
      <div class="bars" :style="{ height: `${height}px` }">
        <div v-for="(row, i) in rows" :key="i" class="bar-wrap"
             :title="`${xLabel(row, i)}: ${fmt(row[m.key])}`">
          <span class="bar-val">{{ fmt(row[m.key]) }}</span>
          <div class="bar" :class="m.cls" :style="{ height: barHeight(row[m.key]) }" />
          <span class="bar-idx">{{ xLabel(row, i) }}</span>
        </div>
        <div v-if="!rows.length" class="empty">暂无数据</div>
      </div>
    </div>
  </div>
</template>

<script setup>
/** 通用柱状趋势图：传入指标定义与数据行即可。 */
const props = defineProps({
  /** [{ key, label, cls, hint }] */
  metrics: { type: Array, default: () => [] },
  /** 数据行数组，按显示顺序传入 */
  rows: { type: Array, default: () => [] },
  /** 每行用于 x 轴显示的字段名 */
  xField: { type: String, default: 'round_index' },
  height: { type: Number, default: 78 },
  /** 数值上限（用于计算柱高），默认按 1 归一化 */
  max: { type: Number, default: 1 },
})

function fmt(v) {
  if (v === null || v === undefined || v === '') return '-'
  const n = Number(v)
  return Number.isNaN(n) ? String(v) : n.toFixed(2)
}

function barHeight(v) {
  const n = Math.max(0, Math.min(props.max, Number(v) || 0))
  return `${Math.max(3, (n / props.max) * 100)}%`
}

function xLabel(row, i) {
  return row?.[props.xField] ?? i + 1
}
</script>

<style scoped>
.trend { display: flex; flex-wrap: wrap; gap: 26px; }
.trend-item { min-width: 200px; }
.trend-label {
  display: flex; align-items: baseline; gap: 6px;
  font-size: 12px; color: var(--text-2); margin-bottom: 6px;
}
.trend-label .hint { font-size: 11px; }
.bars {
  display: flex; align-items: flex-end; gap: 8px;
  position: relative; padding-top: 14px;
}
.bar-wrap {
  display: flex; flex-direction: column; align-items: center;
  justify-content: flex-end; height: 100%; width: 30px;
}
.bar {
  width: 100%; border-radius: 3px 3px 0 0;
  transition: height .3s ease;
  min-height: 3px;
}
.bar-val { font-size: 10px; color: var(--text-3); margin-bottom: 2px; }
.bar-idx { font-size: 10px; color: var(--text-3); margin-top: 3px; }
.empty { font-size: 12px; color: var(--text-3); }

/* 配色 */
.bar.blue { background: #60a5fa; }
.bar.green { background: #34d399; }
.bar.red { background: #f87171; }
.bar.amber { background: #fbbf24; }
.bar.violet { background: #a78bfa; }
</style>
