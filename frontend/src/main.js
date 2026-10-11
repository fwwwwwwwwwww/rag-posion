import { createApp } from 'vue'
import { createRouter, createWebHashHistory } from 'vue-router'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import ChatView from './views/ChatView.vue'
import ExperimentsView from './views/ExperimentsView.vue'
import PoisonDocsView from './views/PoisonDocsView.vue'
import LogsView from './views/LogsView.vue'

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', redirect: '/chat' },
    { path: '/chat', component: ChatView, meta: { title: '问答演示' } },
    { path: '/experiments', component: ExperimentsView, meta: { title: '实验管理' } },
    { path: '/poison-docs', component: PoisonDocsView, meta: { title: '投毒文档' } },
    { path: '/logs', component: LogsView, meta: { title: '攻击日志' } },
  ],
})

createApp(App).use(router).use(ElementPlus).mount('#app')
