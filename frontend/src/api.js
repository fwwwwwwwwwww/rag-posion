/** 统一 API 客户端。
 *
 * 鉴权方式：
 *   - 浏览器走 **HttpOnly Cookie 会话**：登录后由服务端写入，JS 读不到，
 *     因此令牌不会被 XSS 偷走。**前端不保存任何令牌到 localStorage。**
 *   - `withCredentials: true` 让开发时跨端口也能带上 Cookie。
 *   - 需要脚本调用时可用 VITE_API_TOKEN（可选）。
 *
 * 会话失效：拦截 401 并广播事件，由路由守卫统一跳登录页。
 */
import axios from 'axios'

const SCRIPT_TOKEN = import.meta.env.VITE_API_TOKEN || ''

const http = axios.create({
  baseURL: '/api',
  timeout: 120000,
  withCredentials: true, // 关键：带上 HttpOnly 会话 Cookie
})

http.interceptors.request.use((config) => {
  if (SCRIPT_TOKEN) config.headers.Authorization = `Bearer ${SCRIPT_TOKEN}`
  return config
})

const AUTH_EVENT = 'rag:unauthorized'

export function onUnauthorized(handler) {
  window.addEventListener(AUTH_EVENT, handler)
  return () => window.removeEventListener(AUTH_EVENT, handler)
}

function emitUnauthorized(detail) {
  window.dispatchEvent(new CustomEvent(AUTH_EVENT, { detail }))
}

http.interceptors.response.use(
  (resp) => resp.data,
  (err) => {
    const status = err.response?.status
    const data = err.response?.data
    const detail = data?.detail || err.message
    const e = new Error(typeof detail === 'string' ? detail : '请求失败')
    e.status = status
    e.code = data?.code
    if (status === 401 && (data?.code === 'session_expired' || data?.code === 'unauthenticated')) {
      emitUnauthorized({ code: data.code })
    }
    return Promise.reject(e)
  },
)

// ---------------- 鉴权 ----------------
export const auth = {
  login: (username, password) => http.post('/auth/login', { username, password }),
  logout: () => http.post('/auth/logout'),
  me: () => http.get('/auth/me'),
  changePassword: (oldPassword, newPassword) =>
    http.post('/auth/change-password', {
      old_password: oldPassword,
      new_password: newPassword,
    }),
  users: () => http.get('/auth/users'),
  createUser: (data) => http.post('/auth/users', data),
}

// ---------------- SSE 流式请求 ----------------
function sseStream(url, payload, handlers = {}) {
  const { onEvent, onError } = handlers
  return (async () => {
    try {
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include', // 带上会话 Cookie
        body: JSON.stringify(payload),
      })
      if (resp.status === 401) {
        emitUnauthorized({ code: 'session_expired' })
        throw new Error('登录状态已失效，请重新登录')
      }
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const frames = buf.split('\n\n')
        buf = frames.pop()
        for (const frame of frames) {
          const event = (frame.match(/^event: (.+)$/m) || [])[1]
          const dataLine = (frame.match(/^data: (.+)$/m) || [])[1]
          if (!event || !dataLine) continue
          let data
          try {
            data = JSON.parse(dataLine)
          } catch {
            continue
          }
          if (event === 'error') onError?.(data.message)
          else onEvent?.(event, data)
        }
      }
    } catch (e) {
      onError?.(e.message)
    }
  })()
}

export async function streamChat(payload, { onSources, onToken, onDone, onError }) {
  return sseStream('/api/chat/stream', payload, {
    onEvent: (event, data) => {
      if (event === 'sources') onSources?.(data.sources)
      else if (event === 'token') onToken?.(data.text)
      else if (event === 'done') onDone?.(data)
    },
    onError,
  })
}

/** Agent 流式问答：逐步推送决策轨迹（tool_call / tool_result / finalize） */
export async function streamAgentChat(payload, handlers = {}) {
  const { onSession, onStep, onSources, onDone, onError } = handlers
  return sseStream('/api/agent/chat', payload, {
    onEvent: (event, data) => {
      if (event === 'session') onSession?.(data.session_id)
      else if (event === 'step') onStep?.(data)
      else if (event === 'sources') onSources?.(data.sources)
      else if (event === 'done') onDone?.(data)
    },
    onError,
  })
}

export const api = {
  health: () => http.get('/health'),
  attacks: () => http.get('/attacks'),
  logs: (params) => http.get('/logs', { params }),

  createExperiment: (data, autoStart) =>
    http.post('/experiments', data, { params: { auto_start: autoStart } }),
  listExperiments: (params) => http.get('/experiments', { params }),
  getExperiment: (id) => http.get(`/experiments/${id}`),
  getExperimentResults: (id) => http.get(`/experiments/${id}/results`),
  startExperiment: (id) => http.post(`/experiments/${id}/start`),
  compareExperiments: (ids) =>
    http.post('/experiments/compare', { experiment_ids: ids }),

  generatePoison: (data) => http.post('/poison-docs/generate', data),
  listPoison: (params) => http.get('/poison-docs', { params }),
  deletePoison: (id) => http.delete(`/poison-docs/${id}`),
  deletePoisonByAttack: (attackIndex) =>
    http.delete(`/poison-docs/by-attack/${attackIndex}`),

  agentStatus: () => http.get('/agent/status'),
  agentTools: () => http.get('/agent/tools'),
  agentAsk: (data) => http.post('/agent/ask', data),
  resetAgentSession: (sessionId) => http.delete(`/agent/session/${sessionId}`),

  attackStrategies: () => http.get('/attacks/strategies'),
  attackVariants: () => http.get('/attacks/variants'),
  attackIterations: (params) => http.get('/attacks/iterations', { params }),
  startCampaign: (data) => http.post('/attacks/campaign', data),
}
