import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import 'element-plus/dist/index.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

import App from './App.vue'
import router from './router'
import { onUnauthorized } from './api'
import { clearUser } from './stores/auth'
import './styles/global.css'

const app = createApp(App)

// 全局注册图标：模板里可直接 <el-icon><Aim /></el-icon>
for (const [name, comp] of Object.entries(ElementPlusIconsVue)) {
  app.component(name, comp)
}

app.use(router)
app.use(ElementPlus, { locale: zhCn })   // 中文文案

// 会话失效：清本地状态并跳登录页（记住原地址，登录后回跳）
onUnauthorized(() => {
  clearUser()
  const current = router.currentRoute.value
  if (!current.meta.public) {
    router.replace({ name: 'login', query: { redirect: current.fullPath } })
  }
})

app.mount('#app')
