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

async function request(path, options = {}) {
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {})
  }
  const token = getToken()
  if (token) {
    headers.Authorization = `Bearer ${token}`
  }

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
  }
}
