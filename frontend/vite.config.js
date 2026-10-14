import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发时 /api 代理到 FastAPI；生产构建产物由 FastAPI 静态托管
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        // 手动分包：把体积大且很少变动的第三方库单独成块。
        // 好处：业务代码改动后，用户不必重新下载整个 vendor（利用浏览器缓存）。
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined
          if (id.includes('element-plus')) return 'vendor-element'
          if (id.includes('@element-plus/icons')) return 'vendor-icons'
          if (id.includes('/vue/') || id.includes('vue-router') ||
              id.includes('@vue/')) return 'vendor-vue'
          if (id.includes('axios')) return 'vendor-axios'
          return 'vendor'
        },
      },
    },
  },
})
