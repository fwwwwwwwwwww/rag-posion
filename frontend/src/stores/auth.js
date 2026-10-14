/** 登录状态：单一数据源，供路由守卫与布局共用。
 *
 * 注意：这里**不保存任何令牌**——会话在服务端，浏览器只有 HttpOnly Cookie。
 * 前端只缓存"当前用户信息"，用于渲染与权限判断。
 */
import { reactive, readonly } from 'vue'
import { auth } from '../api'

const state = reactive({
  user: null,
  checked: false,   // 是否已向后端确认过登录态
  loading: false,
})

export const authState = readonly(state)

/** 向后端确认当前登录态（刷新页面后调用） */
export async function fetchMe() {
  state.loading = true
  try {
    state.user = await auth.me()
  } catch (e) {
    state.user = null
  } finally {
    state.loading = false
    state.checked = true
  }
  return state.user
}

export async function login(username, password) {
  const user = await auth.login(username, password)
  state.user = user
  state.checked = true
  return user
}

export async function logout() {
  try {
    await auth.logout()
  } catch (e) {
    /* 即使请求失败也要清本地状态 */
  }
  state.user = null
}

/** 会话失效时由 api 层广播，这里只清本地状态 */
export function clearUser() {
  state.user = null
  state.checked = true
}

export function isAdmin() {
  return state.user?.role === 'admin'
}
