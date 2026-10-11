/** 统一 API 客户端：自动携带 Bearer Token（localStorage 可改）。 */
import axios from 'axios'

const TOKEN_KEY = 'rag_auth_token'

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || 'rag-demo-token'
}

export function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token)
}

const http = axios.create({ baseURL: '/api', timeout: 120000 })

http.interceptors.request.use((config) => {
  config.headers.Authorization = `Bearer ${getToken()}`
  return config
})

http.interceptors.response.use(
  (resp) => resp.data,
  (err) => {
    const detail = err.response?.data?.detail || err.message
    return Promise.reject(new Error(detail))
  },
)

// SSE 流式问答（fetch 以便携带鉴权头）
export async function streamChat(payload, { onSources, onToken, onDone, onError }) {
  try {
    const resp = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${getToken()}` },
      body: JSON.stringify(payload),
    })
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
        const data = JSON.parse(dataLine)
        if (event === 'sources') onSources?.(data.sources)
        else if (event === 'token') onToken?.(data.text)
        else if (event === 'done') onDone?.()
        else if (event === 'error') onError?.(data.message)
      }
    }
  } catch (e) {
    onError?.(e.message)
  }
}

export const api = {
  health: () => http.get('/health'),
  attacks: () => http.get('/attacks'),
  logs: (params) => http.get('/logs', { params }),

  createExperiment: (data, autoStart) => http.post('/experiments', data, { params: { auto_start: autoStart } }),
  listExperiments: (params) => http.get('/experiments', { params }),
  getExperiment: (id) => http.get(`/experiments/${id}`),
  getExperimentResults: (id) => http.get(`/experiments/${id}/results`),
  startExperiment: (id) => http.post(`/experiments/${id}/start`),
  compareExperiments: (ids) => http.post('/experiments/compare', { experiment_ids: ids }),

  generatePoison: (data) => http.post('/poison-docs/generate', data),
  listPoison: (params) => http.get('/poison-docs', { params }),
  deletePoison: (id) => http.delete(`/poison-docs/${id}`),
  deletePoisonByAttack: (attackIndex) => http.delete(`/poison-docs/by-attack/${attackIndex}`),
}
