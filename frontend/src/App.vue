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

        <label v-if="authMode === 'register'">
          显示名称
          <input v-model.trim="authForm.displayName" placeholder="例如：张同学" />
        </label>
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
        <button
          v-for="conversation in conversations"
          :key="conversation.id"
          class="conversation-item"
          :class="{ active: activeConversation?.id === conversation.id }"
          @click="selectConversation(conversation.id)"
        >
          <span class="conversation-title">{{ conversation.title }}</span>
          <span class="conversation-meta">{{ modeLabel(conversation.mode) }}</span>
          <span v-if="conversation.last_message" class="conversation-preview">{{ conversation.last_message }}</span>
        </button>
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
        <div class="mode-switch">
          <button :class="{ active: activeMode === 'specialist' }" @click="setMode('specialist')">专科路由</button>
          <button :class="{ active: activeMode === 'general' }" @click="setMode('general')">全科综合</button>
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
            <p>{{ message.content }}</p>
          </div>
        </article>

        <article v-if="loading && pendingText" class="message assistant pending">
          <div class="message-avatar">评</div>
          <div class="message-body">
            <div class="message-role">健康评估建议</div>
            <p>正在结合对话上下文和知识库进行评估...</p>
          </div>
        </article>
      </div>

      <form class="composer" @submit.prevent="send">
        <textarea
          v-model="draft"
          :disabled="loading || !activeConversation"
          rows="3"
          placeholder="输入健康信息，例如：男，45岁，血压148/95，最近睡眠差、运动少，想知道优先处理什么。"
          @keydown.meta.enter.prevent="send"
          @keydown.ctrl.enter.prevent="send"
        />
        <button class="primary-button" :disabled="loading || !draft.trim() || !activeConversation" type="submit">
          {{ loading ? '评估中...' : '发送评估' }}
        </button>
      </form>
    </section>

    <aside class="insight-panel">
      <section>
        <p class="eyebrow">当前评估模式</p>
        <h3>{{ modeLabel(activeMode) }}</h3>
        <p class="muted">{{ activeMode === 'specialist' ? '先由路由智能体选择相关专科，再汇总成综合建议。' : '不做专科路由，直接从整体健康视角输出评估。' }}</p>
      </section>

      <section v-if="latestMetadata">
        <p class="eyebrow">路由与耗时</p>
        <div class="metric-row"><span>Agent</span><strong>{{ latestMetadata.agent_name }}</strong></div>
        <div class="metric-row"><span>Token</span><strong>{{ latestMetadata.total_tokens || 0 }}</strong></div>
        <div class="metric-row"><span>耗时</span><strong>{{ latestMetadata.reasoning_time_seconds || 0 }}s</strong></div>
        <div v-if="latestMetadata.routed_agent_names?.length" class="tag-list">
          <span v-for="name in latestMetadata.routed_agent_names" :key="name">{{ agentLabel(name) }}</span>
        </div>
      </section>

      <section v-if="knowledgeChunks.length">
        <p class="eyebrow">知识库依据</p>
        <div class="source-list">
          <article v-for="(chunk, index) in knowledgeChunks.slice(0, 5)" :key="`${chunk.source_file}-${index}`">
            <strong>{{ chunk.source_file }}</strong>
            <small>{{ chunk.section_path }}</small>
            <p>{{ chunk.content }}</p>
          </article>
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
const authForm = reactive({ username: '', password: '', displayName: '' })
const conversations = ref([])
const activeConversation = ref(null)
const editableTitle = ref('')
const messages = ref([])
const draft = ref('')
const loading = ref(false)
const errorMessage = ref('')
const messagesEl = ref(null)
const pendingText = ref('')

const starters = [
  '男，45岁，血压148/95，最近睡眠差、运动少，想知道优先处理什么。',
  '女，32岁，最近压力大、暴食、体重上涨，晚上经常醒。',
  '父亲62岁，空腹血糖7.8，血脂偏高，平时久坐，想做综合评估。'
]

const activeMode = computed(() => activeConversation.value?.mode || 'specialist')
const latestMetadata = computed(() => {
  const latestAssistant = [...messages.value].reverse().find((item) => item.role === 'assistant' && item.metadata)
  return latestAssistant?.metadata || null
})
const knowledgeChunks = computed(() => latestMetadata.value?.knowledge_chunks || [])

function modeLabel(mode) {
  return mode === 'general' ? '全科综合' : '专科路由'
}

function agentLabel(name) {
  const labels = {
    sleep_activity_nicotine: '睡眠/活动/尼古丁',
    diet_bmi: '饮食/BMI',
    cardiometabolic_health: '血压/血脂/血糖',
    mental_social_health: '心理/社会因素',
    specialist_summary: '专科总结',
    general_health_overview: '全科综合'
  }
  return labels[name] || name
}

async function submitAuth() {
  errorMessage.value = ''
  loading.value = true
  try {
    const payload = authMode.value === 'login'
      ? await api.login({ username: authForm.username, password: authForm.password })
      : await api.register({
          username: authForm.username,
          password: authForm.password,
          display_name: authForm.displayName || authForm.username
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
    mode: activeConversation.value?.mode || 'specialist'
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

async function setMode(mode) {
  if (!activeConversation.value || activeConversation.value.mode === mode) return
  activeConversation.value = await api.updateConversation(activeConversation.value.id, { mode })
  await loadConversations()
}

async function saveTitle() {
  const title = editableTitle.value.trim()
  if (!activeConversation.value || !title || title === activeConversation.value.title) return
  activeConversation.value = await api.updateConversation(activeConversation.value.id, { title })
  await loadConversations()
}

async function send() {
  if (!draft.value.trim() || !activeConversation.value || loading.value) return
  const content = draft.value.trim()
  draft.value = ''
  pendingText.value = content
  loading.value = true
  errorMessage.value = ''
  messages.value.push({ id: `local-${Date.now()}`, role: 'user', content, metadata: null })
  await scrollToBottom()
  try {
    const result = await api.sendMessage(activeConversation.value.id, {
      content,
      mode: activeMode.value
    })
    activeConversation.value = result.conversation
    editableTitle.value = result.conversation.title
    const localIndex = messages.value.findIndex((item) => String(item.id).startsWith('local-'))
    if (localIndex >= 0) {
      messages.value.splice(localIndex, 1, result.user_message)
    }
    messages.value.push(result.assistant_message)
    await loadConversations()
  } catch (error) {
    messages.value.push({
      id: `error-${Date.now()}`,
      role: 'assistant',
      content: `评估失败：${error.message}`,
      metadata: null
    })
  } finally {
    pendingText.value = ''
    loading.value = false
    await scrollToBottom()
  }
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
