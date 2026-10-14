/** 路由与登录守卫。
 *
 * 守卫逻辑：
 *   - 未登录访问受保护页面 → 跳 /login，并记住原目标（登录后回跳）
 *   - 已登录访问 /login     → 跳首页
 *   - 首次进入先用 /auth/me 确认登录态（HttpOnly Cookie 无法在前端判断）
 */
import { createRouter, createWebHashHistory } from 'vue-router'
import { authState, fetchMe } from './stores/auth'

// 路由懒加载：首屏只加载登录页，其余按需拉取，减小首包体积
const LoginView = () => import('./views/LoginView.vue')
const Layout = () => import('./layouts/MainLayout.vue')

const routes = [
  {
    path: '/login',
    name: 'login',
    component: LoginView,
    meta: { public: true, title: '登录' },
  },
  {
    path: '/',
    component: Layout,
    children: [
      { path: '', redirect: '/chat' },
      {
        path: 'chat',
        name: 'chat',
        component: () => import('./views/ChatView.vue'),
        meta: { title: '问答演示', icon: 'ChatDotRound' },
      },
      {
        path: 'battle',
        name: 'battle',
        component: () => import('./views/BattleView.vue'),
        meta: { title: '攻防对抗', icon: 'Aim' },
      },
      {
        path: 'experiments',
        name: 'experiments',
        component: () => import('./views/ExperimentsView.vue'),
        meta: { title: '实验管理', icon: 'DataAnalysis' },
      },
      {
        path: 'poison-docs',
        name: 'poison-docs',
        component: () => import('./views/PoisonDocsView.vue'),
        meta: { title: '投毒文档', icon: 'Warning' },
      },
      {
        path: 'logs',
        name: 'logs',
        component: () => import('./views/LogsView.vue'),
        meta: { title: '攻击日志', icon: 'Document' },
      },
      {
        path: 'settings',
        name: 'settings',
        component: () => import('./views/SettingsView.vue'),
        meta: { title: '账号设置', icon: 'Setting' },
      },
    ],
  },
  { path: '/:pathMatch(.*)*', redirect: '/chat' },
]

const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

router.beforeEach(async (to) => {
  // 首次导航先确认登录态（HttpOnly Cookie 前端读不到，只能问后端）
  if (!authState.checked) {
    await fetchMe()
  }

  if (to.meta.public) {
    // 已登录就别再看登录页
    return authState.user ? { path: '/chat' } : true
  }

  if (!authState.user) {
    return { name: 'login', query: { redirect: to.fullPath } }
  }
  return true
})

router.afterEach((to) => {
  const base = '医疗 RAG 投毒攻防平台'
  document.title = to.meta.title ? `${to.meta.title} · ${base}` : base
})

export default router
