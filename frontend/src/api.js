const TOKEN_KEY = 'healthcare_agent_token'

export function getToken() {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY)
}

function authHeaders(extra = {}) {
  const headers = { ...extra }
  const token = getToken()
  if (token) {
    headers.Authorization = `Bearer ${token}`
  }
  return headers
}

async function request(path, options = {}) {
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {})
  }
  Object.assign(headers, authHeaders())

  const response = await fetch(path, {
    ...options,
    headers
  })
  const text = await response.text()
  const payload = text ? JSON.parse(text) : null
  if (!response.ok) {
    throw new Error(payload?.detail || '请求失败，请稍后重试')
  }
  return payload
}

export const api = {
  register(data) {
    return request('/api/v1/auth/register', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  },
  login(data) {
    return request('/api/v1/auth/login', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  },
  me() {
    return request('/api/v1/auth/me')
  },
  listConversations() {
    return request('/api/v1/conversations')
  },
  createConversation(data) {
    return request('/api/v1/conversations', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  },
  getConversation(id) {
    return request(`/api/v1/conversations/${id}`)
  },
  updateConversation(id, data) {
    return request(`/api/v1/conversations/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(data)
    })
  },
  deleteConversation(id) {
    return request(`/api/v1/conversations/${id}`, {
      method: 'DELETE'
    })
  },
  sendMessage(id, data) {
    return request(`/api/v1/conversations/${id}/messages`, {
      method: 'POST',
      body: JSON.stringify(data)
    })
  },
  async streamMessage(id, data, onEvent) {
    return streamRequest(`/api/v1/conversations/${id}/messages/stream`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(data)
    }, onEvent)
  },
  async streamFiles(id, formData, onEvent) {
    return streamRequest(`/api/v1/conversations/${id}/messages/files/stream`, {
      method: 'POST',
      headers: authHeaders(),
      body: formData
    }, onEvent)
  }
}

async function streamRequest(path, options, onEvent) {
  const response = await fetch(path, options)
  if (!response.ok) {
    const text = await response.text()
    let detail = '请求失败，请稍后重试'
    try {
      detail = JSON.parse(text)?.detail || detail
    } catch {
      detail = text || detail
    }
    throw new Error(detail)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop() || ''
    for (const part of parts) {
      const line = part.split('\n').find((item) => item.startsWith('data: '))
      if (!line) continue
      onEvent(JSON.parse(line.slice(6)))
    }
  }

  if (buffer.trim().startsWith('data: ')) {
    onEvent(JSON.parse(buffer.trim().slice(6)))
  }
}
