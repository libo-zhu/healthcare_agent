<template>
  <main v-if="!user" class="auth-shell">
    <section class="auth-panel">
      <div class="brand-lockup">
        <div class="brand-mark">H</div>
        <div>
          <p class="eyebrow">Health Navigator</p>
          <h1>个人健康评估智能体</h1>
        </div>
      </div>

      <div class="auth-copy">
        <p>围绕睡眠、运动、饮食、体重、血压、血脂、血糖、心理与社会因素，建立连续健康评估记录。</p>
      </div>

      <form class="auth-card" @submit.prevent="submitAuth">
        <div class="segmented">
          <button type="button" :class="{ active: authMode === 'login' }" @click="authMode = 'login'">登录</button>
          <button type="button" :class="{ active: authMode === 'register' }" @click="authMode = 'register'">注册</button>
        </div>

        <label>
          用户名
          <input v-model.trim="authForm.username" required autocomplete="username" placeholder="请输入用户名" />
        </label>
        <label>
          密码
          <input v-model="authForm.password" required type="password" autocomplete="current-password" placeholder="至少 6 位" />
        </label>

        <p v-if="errorMessage" class="error-text">{{ errorMessage }}</p>
        <button class="primary-button" :disabled="loading" type="submit">
          {{ loading ? '处理中...' : authMode === 'login' ? '进入评估工作台' : '创建账户' }}
        </button>
      </form>
    </section>
  </main>

  <main v-else class="app-shell">
    <aside class="sidebar">
      <div class="sidebar-head">
        <div class="brand-lockup compact">
          <div class="brand-mark">H</div>
          <div>
            <p class="eyebrow">Health Navigator</p>
            <strong>{{ user.display_name }}</strong>
          </div>
        </div>
        <button class="icon-button" title="退出登录" @click="logout">退出</button>
      </div>

      <button class="new-chat-button" @click="createNewConversation">新建健康评估</button>

      <div class="conversation-list">
        <div
          v-for="conversation in conversations"
          :key="conversation.id"
          class="conversation-item"
          :class="{ active: activeConversation?.id === conversation.id }"
          @click="selectConversation(conversation.id)"
        >
          <button class="conversation-main" type="button">
            <span class="conversation-title">{{ conversation.title }}</span>
            <span v-if="conversation.last_message" class="conversation-preview">{{ conversation.last_message }}</span>
          </button>
          <button
            class="conversation-delete"
            type="button"
            title="删除对话"
            @click.stop="deleteConversation(conversation.id)"
          >
            删除
          </button>
        </div>
      </div>
    </aside>

    <section class="chat-surface">
      <header class="chat-header">
        <div>
          <p class="eyebrow">连续健康档案对话</p>
          <input
            v-if="activeConversation"
            v-model="editableTitle"
            class="title-input"
            @blur="saveTitle"
            @keydown.enter.prevent="saveTitle"
          />
        </div>
      </header>

      <div class="health-strip">
        <div><span>睡眠</span><strong>作息与质量</strong></div>
        <div><span>代谢</span><strong>血压血脂血糖</strong></div>
        <div><span>营养</span><strong>饮食与 BMI</strong></div>
        <div><span>心理</span><strong>压力与支持</strong></div>
      </div>

      <div ref="messagesEl" class="messages">
        <div v-if="messages.length === 0" class="empty-state">
          <h2>从一次真实的健康描述开始</h2>
          <p>可以输入体检指标、近期症状、生活方式、心理压力或想追踪的问题。系统会把本次对话历史作为上下文，但医学判断仍以你提供的信息和知识库依据为边界。</p>
          <div class="starter-grid">
            <button v-for="starter in starters" :key="starter" @click="draft = starter">{{ starter }}</button>
          </div>
        </div>

        <article v-for="message in messages" :key="message.id" class="message" :class="message.role">
          <div class="message-avatar">{{ message.role === 'user' ? '我' : '评' }}</div>
          <div class="message-body">
            <div class="message-role">{{ message.role === 'user' ? '我的健康信息' : '健康评估建议' }}</div>
            <p v-if="message.role === 'user'">{{ message.content }}</p>
            <div v-else class="markdown-body" v-html="renderMarkdown(message.content)"></div>
          </div>
        </article>

        <article v-if="loading && pendingText && streamStatus" class="message assistant pending">
          <div class="message-avatar">评</div>
          <div class="message-body">
            <div class="message-role">健康评估建议</div>
            <p>{{ streamStatus || '正在结合对话上下文和知识库进行评估...' }}</p>
          </div>
        </article>
      </div>

      <form class="composer" @submit.prevent="send">
        <div class="composer-main">
          <textarea
            v-model="draft"
            :disabled="loading || !activeConversation"
            rows="3"
            placeholder="输入健康问题，也可以上传病例图片或 PDF，例如：请结合这份体检报告评估血压和血糖风险。"
            @keydown.meta.enter.prevent="send"
            @keydown.ctrl.enter.prevent="send"
          />
          <div class="attachment-row">
            <label class="file-button">
              上传病例
              <input
                type="file"
                multiple
                accept=".pdf,.png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp"
                :disabled="loading || !activeConversation"
                @change="handleFiles"
              />
            </label>
            <span v-if="selectedFiles.length" class="file-summary">
              {{ selectedFiles.map((file) => file.name).join('、') }}
            </span>
            <button v-if="selectedFiles.length" class="clear-files-button" type="button" @click="clearFiles">清空</button>
          </div>
        </div>
        <button class="primary-button" :disabled="loading || (!draft.trim() && !selectedFiles.length) || !activeConversation" type="submit">
          {{ loading ? '评估中...' : '发送评估' }}
        </button>
      </form>
    </section>

    <aside class="insight-panel">
      <section v-if="knowledgeChunks.length">
        <p class="eyebrow">知识库依据</p>
        <div class="source-list">
          <details
            v-for="(source, index) in knowledgeSources"
            :key="`${source.sourceFile}-${index}`"
            class="source-card"
            :open="index === 0"
          >
            <summary>
              <span class="source-index">依据 {{ index + 1 }}</span>
              <strong>{{ source.title }}</strong>
              <small>{{ source.path }}</small>
              <p>{{ source.summary }}</p>
            </summary>
            <div class="source-detail">
              <dl v-if="source.items.length">
                <template v-for="item in source.items" :key="`${source.sourceFile}-${item.label}-${item.value}`">
                  <dt>{{ item.label }}</dt>
                  <dd>{{ item.value }}</dd>
                </template>
              </dl>
              <p v-else>{{ source.fullText }}</p>
            </div>
          </details>
        </div>
      </section>

      <section>
        <p class="eyebrow">安全边界</p>
        <p class="muted">本系统用于健康评估和风险提示，不替代线下医生诊断。出现胸痛、严重呼吸困难、意识异常、自伤风险等情况，应立即寻求急救或专业帮助。</p>
      </section>
    </aside>
  </main>
</template>

<script setup>
import { computed, nextTick, onMounted, reactive, ref } from 'vue'
import { api, clearToken, getToken, setToken } from './api'

const user = ref(null)
const authMode = ref('login')
const authForm = reactive({ username: '', password: '' })
const conversations = ref([])
const activeConversation = ref(null)
const editableTitle = ref('')
const messages = ref([])
const draft = ref('')
const selectedFiles = ref([])
const loading = ref(false)
const errorMessage = ref('')
const messagesEl = ref(null)
const pendingText = ref('')
const streamStatus = ref('')

const starters = [
  '男，45岁，血压148/95，最近睡眠差、运动少，想知道优先处理什么。',
  '女，32岁，最近压力大、暴食、体重上涨，晚上经常醒。',
  '父亲62岁，空腹血糖7.8，血脂偏高，平时久坐，想做综合评估。'
]

const latestMetadata = computed(() => {
  const latestAssistant = [...messages.value].reverse().find((item) => item.role === 'assistant' && item.metadata)
  return latestAssistant?.metadata || null
})
const knowledgeChunks = computed(() => latestMetadata.value?.knowledge_chunks || [])
const knowledgeSources = computed(() => knowledgeChunks.value.slice(0, 6).map(normalizeKnowledgeChunk))

async function submitAuth() {
  errorMessage.value = ''
  loading.value = true
  try {
    const payload = authMode.value === 'login'
      ? await api.login({ username: authForm.username, password: authForm.password })
      : await api.register({
          username: authForm.username,
          password: authForm.password
        })
    setToken(payload.access_token)
    user.value = payload.user
    await loadConversations()
    if (!activeConversation.value) {
      await createNewConversation()
    }
  } catch (error) {
    errorMessage.value = error.message
  } finally {
    loading.value = false
  }
}

async function bootstrap() {
  if (!getToken()) return
  try {
    user.value = await api.me()
    await loadConversations()
    if (conversations.value.length) {
      await selectConversation(conversations.value[0].id)
    } else {
      await createNewConversation()
    }
  } catch {
    clearToken()
    user.value = null
  }
}

async function loadConversations() {
  conversations.value = await api.listConversations()
}

async function createNewConversation() {
  const conversation = await api.createConversation({
    title: '新的健康评估',
    mode: 'specialist'
  })
  await loadConversations()
  await selectConversation(conversation.id)
}

async function selectConversation(id) {
  const detail = await api.getConversation(id)
  activeConversation.value = detail.conversation
  editableTitle.value = detail.conversation.title
  messages.value = detail.messages
  await scrollToBottom()
}

async function deleteConversation(id) {
  if (loading.value) return
  const conversation = conversations.value.find((item) => item.id === id)
  const ok = window.confirm(`确定删除「${conversation?.title || '这段对话'}」吗？删除后无法恢复。`)
  if (!ok) return

  await api.deleteConversation(id)
  await loadConversations()

  if (activeConversation.value?.id === id) {
    activeConversation.value = null
    editableTitle.value = ''
    messages.value = []
    if (conversations.value.length) {
      await selectConversation(conversations.value[0].id)
    } else {
      await createNewConversation()
    }
  }
}

async function saveTitle() {
  const title = editableTitle.value.trim()
  if (!activeConversation.value || !title || title === activeConversation.value.title) return
  activeConversation.value = await api.updateConversation(activeConversation.value.id, { title })
  await loadConversations()
}

async function send() {
  if ((!draft.value.trim() && !selectedFiles.value.length) || !activeConversation.value || loading.value) return
  const content = draft.value.trim()
  const files = [...selectedFiles.value]
  draft.value = ''
  selectedFiles.value = []
  pendingText.value = content || files.map((file) => file.name).join('、')
  loading.value = true
  streamStatus.value = '正在读取输入...'
  errorMessage.value = ''
  const localAssistantId = `assistant-${Date.now()}`
  messages.value.push({
    id: `local-${Date.now()}`,
    role: 'user',
    content: buildVisibleUserContent(content, files),
    metadata: null
  })
  messages.value.push({
    id: localAssistantId,
    role: 'assistant',
    content: '',
    metadata: null
  })
  await scrollToBottom()
  try {
    const onEvent = async (event) => {
      if (event.type === 'rewrite') {
        streamStatus.value = '正在整理健康信息...'
      } else if (event.type === 'route') {
        streamStatus.value = '正在匹配评估方向...'
      } else if (event.type === 'knowledge') {
        streamStatus.value = '正在检索医学知识库依据...'
      } else if (event.type === 'source') {
        streamStatus.value = '病例资料已解析，正在评估...'
      } else if (event.type === 'token') {
        const assistant = messages.value.find((item) => item.id === localAssistantId)
        if (assistant) assistant.content += event.content || ''
        streamStatus.value = ''
      } else if (event.type === 'user_message') {
        const localIndex = messages.value.findIndex((item) => String(item.id).startsWith('local-'))
        if (localIndex >= 0) {
          messages.value.splice(localIndex, 1, {
            id: event.id,
            role: 'user',
            content: event.content,
            metadata: null
          })
        }
      } else if (event.type === 'persisted') {
        activeConversation.value = event.conversation
        editableTitle.value = event.conversation.title
        const assistantIndex = messages.value.findIndex((item) => item.id === localAssistantId)
        if (assistantIndex >= 0) {
          messages.value.splice(assistantIndex, 1, event.assistant_message)
        }
      } else if (event.type === 'error') {
        throw new Error(event.content || '评估失败')
      }
      await scrollToBottom()
    }

    if (files.length) {
      const formData = new FormData()
      formData.append('medical_data', content)
      formData.append('mode', 'specialist')
      for (const file of files) {
        formData.append('files', file)
      }
      await api.streamFiles(activeConversation.value.id, formData, onEvent)
    } else {
      await api.streamMessage(activeConversation.value.id, {
        content,
        mode: 'specialist'
      }, onEvent)
    }
    await loadConversations()
  } catch (error) {
    const assistant = messages.value.find((item) => item.id === localAssistantId)
    if (assistant) {
      assistant.content = `评估失败：${error.message}`
    }
  } finally {
    pendingText.value = ''
    streamStatus.value = ''
    loading.value = false
    await scrollToBottom()
  }
}

function handleFiles(event) {
  selectedFiles.value = Array.from(event.target.files || [])
  event.target.value = ''
}

function clearFiles() {
  selectedFiles.value = []
}

function buildVisibleUserContent(content, files) {
  const parts = []
  if (content) parts.push(content)
  if (files.length) parts.push(`上传资料：${files.map((file) => file.name).join('、')}`)
  return parts.join('\n')
}

function normalizeKnowledgeChunk(chunk) {
  const sourceFile = chunk.source_file || '知识库依据'
  const title = stripJsonExtension(sourceFile)
  const items = parseKnowledgeItems(chunk.content || '')
  const path = formatKnowledgePath(chunk.section_path, items)
  const summary = buildKnowledgeSummary(items, chunk.content)
  return {
    sourceFile,
    title,
    path,
    summary,
    items,
    fullText: chunk.content || ''
  }
}

function stripJsonExtension(value) {
  return String(value).replace(/\.json$/i, '')
}

function parseKnowledgeItems(content) {
  return String(content || '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const match = line.match(/^(.+?):\s*(.*)$/)
      if (!match) return { label: '内容', value: line }
      return {
        label: formatKnowledgeKey(match[1]),
        value: match[2]
      }
    })
    .filter((item) => item.value)
}

function formatKnowledgeKey(value) {
  const cleaned = String(value)
    .replace(/^root\.?/, '')
    .replace(/children\[\d+\]\.?/g, '')
    .replace(/\.+/g, '.')
    .replace(/^\./, '')

  const labels = {
    topic: '主题',
    content: '依据内容'
  }
  return labels[cleaned] || cleaned || '内容'
}

function formatKnowledgePath(sectionPath, items) {
  const topics = items
    .filter((item) => item.label === '主题')
    .map((item) => item.value)
  if (topics.length) return topics.join(' / ')

  const cleaned = String(sectionPath || '')
    .replace(/^root\.?/, '')
    .replace(/children\[(\d+)\]/g, '第 $1 节')
    .replace(/\./g, ' / ')
    .trim()
  return cleaned || '相关知识片段'
}

function buildKnowledgeSummary(items, fallback) {
  const contentItem = items.find((item) => item.label === '依据内容')
  const raw = contentItem?.value || String(fallback || '')
  const compact = raw.replace(/\s+/g, ' ').trim()
  return compact.length > 94 ? `${compact.slice(0, 94)}...` : compact
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
}

function renderInlineMarkdown(value) {
  return value
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+?)`/g, '<code>$1</code>')
}

function renderMarkdown(value) {
  const escaped = escapeHtml(value || '')
  const lines = escaped.split('\n')
  const html = []
  let inList = false

  for (const rawLine of lines) {
    const line = rawLine.trim()
    if (!line) {
      if (inList) {
        html.push('</ul>')
        inList = false
      }
      continue
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/)
    if (heading) {
      if (inList) {
        html.push('</ul>')
        inList = false
      }
      html.push(`<h${heading[1].length}>${renderInlineMarkdown(heading[2])}</h${heading[1].length}>`)
      continue
    }
    const bullet = line.match(/^[-*]\s+(.+)$/)
    if (bullet) {
      if (!inList) {
        html.push('<ul>')
        inList = true
      }
      html.push(`<li>${renderInlineMarkdown(bullet[1])}</li>`)
      continue
    }
    if (inList) {
      html.push('</ul>')
      inList = false
    }
    html.push(`<p>${renderInlineMarkdown(line)}</p>`)
  }

  if (inList) html.push('</ul>')
  return html.join('')
}

function logout() {
  clearToken()
  user.value = null
  conversations.value = []
  activeConversation.value = null
  messages.value = []
}

async function scrollToBottom() {
  await nextTick()
  if (messagesEl.value) {
    messagesEl.value.scrollTop = messagesEl.value.scrollHeight
  }
}

onMounted(bootstrap)
</script>
