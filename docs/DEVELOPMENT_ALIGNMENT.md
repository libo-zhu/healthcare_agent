# Healthcare Agent Development Alignment

这份文档用于统一 agent 系统、后端 API 和后续前端网页开发之间的理解。后续改后端或做前端时，优先以这里的契约为准；如果代码行为变化，应同步更新本文件。

## 1. 系统边界

当前仓库是 Python 后端与 agent 核心实现，尚未包含独立前端工程。

核心能力：

- 接收用户健康文本，或接收文本 + 文件上传内容。
- 支持用户注册、登录、会话列表、新建会话和对话历史持久化。
- 支持在会话内把最近历史消息拼入当前 agent 输入，实现轻量上下文记忆。
- 先用 query rewrite agent 整理输入。
- 专科流程会经过 router agent，选择 1 到 4 个专科 agent 并行分析，再由 summary agent 汇总。
- 全科流程不走专科 router，直接由 general agent 输出整体健康评估。
- RAG 从本地 JSON 知识库构建 Chroma 向量索引，先粗召回，再用 reranker 重排。
- 普通接口返回完整 JSON；流式接口返回 SSE 事件，适合网页逐步渲染。

运行时主要组件：

- `src/healthcare_agent/api.py`: FastAPI 路由和 HTTP 契约。
- `src/healthcare_agent/agent.py`: agent 编排、流式事件生成、token 统计聚合。
- `src/healthcare_agent/prompts.py`: router、rewrite、专科、全科、总结 prompt。
- `src/healthcare_agent/knowledge_base.py`: JSON 知识库索引、检索、重排、知识片段格式化。
- `src/healthcare_agent/preprocessing.py`: 上传文件解析、PDF 文本提取、图片 OCR。
- `src/healthcare_agent/schemas.py`: 请求和响应 Pydantic schema。
- `src/healthcare_agent/database.py`: MySQL 连接和表结构初始化。
- `src/healthcare_agent/auth.py`: 密码哈希、Bearer token 和当前用户解析。
- `src/healthcare_agent/chat_service.py`: 会话、消息和上下文评估服务。
- `frontend/`: Vue 健康评估网页。
- `evaluate_test_data.py`: 批量评测脚本。
- `judge_manual_review.py`: DeepSeek 作为 judge 的半自动评分脚本。

## 2. 环境与启动

Python 版本要求：`>=3.12`。

安装：

```bash
pip install -e .
```

环境变量写在项目根目录 `.env`。可从 `.env.example` 复制：

```bash
cp .env.example .env
```

关键配置：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 无 | 必填，DeepSeek API key |
| `AUTH_SECRET_KEY` | 无 | 建议设置，用于签发登录 token |
| `MYSQL_HOST` | `127.0.0.1` | MySQL 地址 |
| `MYSQL_PORT` | `3306` | MySQL 端口 |
| `MYSQL_USER` | `root` | MySQL 用户 |
| `MYSQL_PASSWORD` | `zhu203926` | 本地 MySQL 密码 |
| `MYSQL_DATABASE` | `healthcare_agent_app` | 应用数据库名 |
| `RAG_ENABLED` | `true` | 是否启用 RAG |
| `RAG_AUTO_BUILD` | `true` | 检索时发现索引为空是否自动构建 |
| `RAG_TOP_K` | `5` | 兼容参数；当前主流程未显式使用，默认以粗召回和重排配置为准 |
| `RAG_COARSE_TOP_K` | `20` | Chroma 粗召回数量 |
| `RAG_RERANK_TOP_K` | `5` | reranker 后最终注入 prompt 的片段数 |
| `MEDICAL_KNOWLEDGE_BASE_DIR` | `knowledge_base` | 本地 JSON 知识库目录 |
| `CHROMA_PERSIST_DIR` | `.chroma/medical_kb` | Chroma 持久化目录 |
| `VECTOR_COLLECTION_NAME` | `medical_knowledge` | Chroma collection 名称 |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-zh-v1.5` | 向量模型或本地模型路径 |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-base` | CrossEncoder reranker 模型或本地路径 |

构建或重建知识库索引：

```bash
python main.py --rebuild-kb
```

查看知识库状态：

```bash
python main.py --kb-status
```

启动 API：

```bash
python run_api.py
```

默认服务地址：`http://127.0.0.1:8000`。Swagger：`http://127.0.0.1:8000/docs`。

开发阶段 CORS 已开放：允许任意来源、请求头和方法。前端本地页面可以直接请求该 API。

启动前端：

```bash
cd frontend
npm install
npm run dev
```

默认前端地址：`http://localhost:5173/`。Vite 已配置 `/api` 代理到 `http://127.0.0.1:8000`。

## 3. Agent 流程

### 3.1 专科流程

入口：

- `POST /api/v1/specialist/assessment`
- `POST /api/v1/specialist/assessment/stream`
- `POST /api/v1/specialist/assessment/files`
- `POST /api/v1/specialist/assessment/files/stream`

执行顺序：

1. `rewrite_medical_data`: 将口语化输入、PDF 提取文本、OCR 文本整理成清晰医疗输入。
2. `route_to_specialists`: 返回 `agent_names` 和 `reason`。
3. `run_parallel_specialist_assessments`: 对被选中的专科并行检索知识库和调用模型。
4. `summarize_specialist_assessments`: 汇总多个专科结果，形成最终 `content`。
5. 聚合 token、耗时、知识片段和专科结果。

专科 agent 名称固定为：

| agent_name | 前端展示名 | 范围 |
| --- | --- | --- |
| `sleep_activity_nicotine` | `Sleep / Activity / Nicotine` | 睡眠、身体活动、久坐、运动、吸烟、二手烟、电子烟 |
| `diet_bmi` | `Diet / BMI` | 饮食、营养、BMI、体重管理、减重、消瘦 |
| `cardiometabolic_health` | `Blood Pressure / Lipids / Glucose` | 血压、血脂、血糖、心代谢风险 |
| `mental_social_health` | `Mental Health / SDOH` | 心理压力、情绪、社会决定因素 |

专科普通接口最终响应的 `agent_name` 不是某个专科，而是 `specialist_summary`。真正参与的专科在 `routed_agent_names` 和 `specialist_assessments` 中。

### 3.2 全科流程

入口：

- `POST /api/v1/general/assessment`
- `POST /api/v1/general/assessment/stream`
- `POST /api/v1/general/assessment/files`
- `POST /api/v1/general/assessment/files/stream`

执行顺序：

1. `rewrite_medical_data`
2. 使用整理后的输入检索全部知识库，不按专科过滤。
3. 调用 `general_health_overview` agent。

全科响应中：

- `agent_name`: `general_health_overview`
- `routed_agent_names`: `["general_health_overview"]`
- `route_reason`: `direct generalist assessment`
- `specialist_assessments`: 空数组

## 4. HTTP API 契约

根路径：

```http
GET /
```

返回服务状态和接口路径索引。

### 4.0 登录、注册与会话接口

注册：

```http
POST /api/v1/auth/register
Content-Type: application/json
```

```json
{
  "username": "demo",
  "password": "zhu203926",
  "display_name": "测试用户"
}
```

登录：

```http
POST /api/v1/auth/login
Content-Type: application/json
```

```json
{
  "username": "demo",
  "password": "zhu203926"
}
```

认证响应：

```json
{
  "access_token": "...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "username": "demo",
    "display_name": "测试用户",
    "created_at": "2026-04-26 23:00:43"
  }
}
```

后续会话接口都需要请求头：

```http
Authorization: Bearer <access_token>
```

当前用户：

```http
GET /api/v1/auth/me
```

会话列表：

```http
GET /api/v1/conversations
```

新建会话：

```http
POST /api/v1/conversations
Content-Type: application/json
```

```json
{
  "title": "新的健康评估",
  "mode": "specialist"
}
```

读取会话详情和消息历史：

```http
GET /api/v1/conversations/{conversation_id}
```

更新会话标题或模式：

```http
PATCH /api/v1/conversations/{conversation_id}
Content-Type: application/json
```

```json
{
  "title": "血压和睡眠跟踪",
  "mode": "general"
}
```

删除会话：

```http
DELETE /api/v1/conversations/{conversation_id}
```

在会话内发送一轮健康评估：

```http
POST /api/v1/conversations/{conversation_id}/messages
Content-Type: application/json
```

```json
{
  "content": "男，45岁，血压148/95，最近睡眠差、运动少，想知道优先处理什么。",
  "mode": "specialist"
}
```

响应会返回本轮保存后的用户消息和助手消息：

```json
{
  "conversation": {
    "id": 1,
    "title": "男，45岁，血压148/95，最近睡眠差",
    "mode": "specialist",
    "created_at": "2026-04-26 23:00:43",
    "updated_at": "2026-04-26 23:02:10",
    "last_message": null
  },
  "user_message": {
    "id": 1,
    "role": "user",
    "content": "男，45岁，血压148/95...",
    "metadata": null,
    "created_at": null
  },
  "assistant_message": {
    "id": 2,
    "role": "assistant",
    "content": "最终健康评估建议",
    "metadata": {
      "mode": "specialist",
      "agent_name": "specialist_summary",
      "routed_agent_names": ["cardiometabolic_health"],
      "knowledge_chunks": []
    },
    "created_at": null
  }
}
```

上下文记忆规则：

- 后端会读取当前会话最近 10 条历史消息。
- 历史消息会作为“同一用户当前健康评估对话的背景”拼入最新输入。
- 原 agent 仍保持单轮接口；上下文能力由 `chat_service.py` 在调用前完成。
- 用户原始消息和助手最终回答都会写入 `messages` 表。
- 助手消息的 `metadata` 保存路由、token、耗时、知识依据和专科结果，供前端侧栏展示。

### 4.1 文本评估接口

```http
POST /api/v1/specialist/assessment
POST /api/v1/general/assessment
Content-Type: application/json
```

请求体：

```json
{
  "medical_data": "患者，45岁，男性，血压148/95 mmHg，空腹血糖6.8 mmol/L，最近经常熬夜。"
}
```

字段要求：

- `medical_data`: 必填字符串，长度至少 1。

响应体统一使用 `AssessmentResponse`：

```json
{
  "content": "最终回答文本",
  "agent_name": "specialist_summary",
  "routed_agent_names": ["cardiometabolic_health", "sleep_activity_nicotine"],
  "route_reason": "路由理由",
  "rewritten_query": "整理后的输入",
  "input_tokens": 123,
  "output_tokens": 456,
  "total_tokens": 579,
  "reasoning_time_seconds": 12.345,
  "source_summary": "inline_text",
  "preprocessed_text": "原始输入文本",
  "preprocessing_notes": [],
  "knowledge_chunks": [],
  "coarse_knowledge_chunks": [],
  "reranked_knowledge_chunks": [],
  "specialist_assessments": []
}
```

重要字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `content` | string | 前端主回答区应展示的最终回答 |
| `agent_name` | string | 最终回答 agent；专科流程为 `specialist_summary` |
| `routed_agent_names` | string[] | 实际参与分析的 agent 名称 |
| `route_reason` | string | router 或流程说明 |
| `rewritten_query` | string/null | rewrite agent 整理后的输入 |
| `source_summary` | string/null | 输入来源摘要，文本接口固定为 `inline_text` |
| `preprocessed_text` | string/null | 最终送入 agent 的预处理文本 |
| `preprocessing_notes` | string[] | 文件解析或 OCR 的提示信息 |
| `knowledge_chunks` | KnowledgeChunk[] | 最终注入 prompt 的知识片段 |
| `coarse_knowledge_chunks` | KnowledgeChunk[] | 向量粗召回片段 |
| `reranked_knowledge_chunks` | KnowledgeChunk[] | reranker 后片段；当前等同最终知识片段 |
| `specialist_assessments` | SpecialistAssessment[] | 专科流程下每个专科的独立结果 |

`KnowledgeChunk`：

```json
{
  "source_file": "血压.json",
  "section_path": "root.assessment_items[0]",
  "content": "知识片段正文",
  "score": 0.87,
  "vector_score": 0.76,
  "rerank_score": 0.87
}
```

`SpecialistAssessment`：

```json
{
  "agent_name": "cardiometabolic_health",
  "agent_label": "Blood Pressure / Lipids / Glucose",
  "content": "该专科的独立分析",
  "usage": {
    "input_tokens": 100,
    "output_tokens": 200,
    "total_tokens": 300
  },
  "reasoning_time_seconds": 5.123,
  "knowledge_chunks": [],
  "coarse_knowledge_chunks": [],
  "reranked_knowledge_chunks": []
}
```

注意：普通 JSON 响应中的 `specialist_assessments[].usage` 是嵌套对象；流式 `specialist_result` 事件为了前端增量处理方便，使用扁平的 `input_tokens`、`output_tokens`、`total_tokens` 字段。

### 4.2 文件上传评估接口

```http
POST /api/v1/specialist/assessment/files
POST /api/v1/general/assessment/files
Content-Type: multipart/form-data
```

表单字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `medical_data` | Text | 否 | 补充说明文字 |
| `files` | File[] | 否 | 一个或多个上传文件 |

必须至少提供 `medical_data` 或 `files` 之一。

支持文件：

- 文本：`.txt`
- 表格文本：`.csv`
- PDF：`.pdf`
- 图片 OCR：`.png`、`.jpg`、`.jpeg`、`.bmp`、`.tif`、`.tiff`、`.webp`

预处理规则：

- 文本输入会原样去首尾空白。
- 每个文件会被拼接成：

```text
文件名: report.pdf
提取内容:
...
```

- 多个来源之间用两个换行拼接。
- `source_summary` 是 `inline_text` 和文件名的逗号分隔列表。
- 图片 OCR 依赖系统级 `tesseract`；缺少时图片接口会返回 500，文本/PDF/CSV 不受影响。

文件上传普通接口响应仍是 `AssessmentResponse`，但 `source_summary`、`preprocessed_text`、`preprocessing_notes` 对前端调试很重要，建议提供可展开的“输入解析结果”面板。

### 4.3 SSE 流式接口

```http
POST /api/v1/specialist/assessment/stream
POST /api/v1/general/assessment/stream
Content-Type: application/json
Accept: text/event-stream
```

```http
POST /api/v1/specialist/assessment/files/stream
POST /api/v1/general/assessment/files/stream
Content-Type: multipart/form-data
Accept: text/event-stream
```

所有事件均是 SSE `data:` 行，内容为 JSON：

```text
data: {"type":"token","content":"..."}

```

前端需要按空行分隔事件，并对每条 `data:` 的 JSON 做解析。

通用事件：

| type | 出现接口 | 说明 |
| --- | --- | --- |
| `source` | 文件流式接口 | 文件预处理完成后第一个事件 |
| `rewrite` | 全部流式接口 | query rewrite 完成 |
| `route` | 全部流式接口 | 专科路由或全科直接流程说明 |
| `knowledge` | 全部流式接口 | 某个 agent 的知识片段 |
| `specialist_result` | 仅专科流式接口 | 单个专科完成后的独立分析 |
| `token` | 全部流式接口 | 最终回答的增量 token |
| `done` | 全部流式接口 | 流式完成和最终元数据 |
| `error` | 全部流式接口 | 执行错误 |

文件流式接口的 `source` 事件：

```json
{
  "type": "source",
  "source_summary": "inline_text, report.pdf",
  "preprocessed_text": "最终送入 agent 的文本",
  "preprocessing_notes": ["OCR languages used: eng+chi_sim"]
}
```

`rewrite` 事件：

```json
{
  "type": "rewrite",
  "rewritten_query": "整理后的输入"
}
```

专科 `route` 事件：

```json
{
  "type": "route",
  "agent_name": "specialist_summary",
  "agent_label": "Specialist Summary",
  "agent_names": ["cardiometabolic_health"],
  "agent_labels": ["Blood Pressure / Lipids / Glucose"],
  "reason": "简短路由理由"
}
```

全科 `route` 事件：

```json
{
  "type": "route",
  "agent_name": "general_health_overview",
  "agent_label": "General Health Overview",
  "reason": "direct generalist assessment"
}
```

`knowledge` 事件：

```json
{
  "type": "knowledge",
  "agent_name": "cardiometabolic_health",
  "agent_label": "Blood Pressure / Lipids / Glucose",
  "chunks": [],
  "coarse_chunks": [],
  "reranked_chunks": []
}
```

注意：全科 `knowledge` 事件当前没有 `agent_name` 和 `agent_label` 字段；前端应使用当前流程上下文兜底展示为 `general_health_overview`。

`specialist_result` 事件：

```json
{
  "type": "specialist_result",
  "agent_name": "cardiometabolic_health",
  "agent_label": "Blood Pressure / Lipids / Glucose",
  "content": "该专科完整分析",
  "input_tokens": 100,
  "output_tokens": 200,
  "total_tokens": 300,
  "reasoning_time_seconds": 5.123
}
```

`token` 事件：

```json
{
  "type": "token",
  "content": "增量文本"
}
```

`done` 事件：

专科流式：

```json
{
  "type": "done",
  "agent_name": "specialist_summary",
  "agent_label": "Specialist Summary",
  "routed_agent_names": ["cardiometabolic_health"],
  "rewritten_query": "整理后的输入",
  "content": "完整最终回答",
  "knowledge_chunks": [],
  "coarse_knowledge_chunks": [],
  "reranked_knowledge_chunks": [],
  "input_tokens": 123,
  "output_tokens": 456,
  "total_tokens": 579,
  "reasoning_time_seconds": 12.345
}
```

全科流式：

```json
{
  "type": "done",
  "agent_name": "general_health_overview",
  "rewritten_query": "整理后的输入",
  "input_tokens": 123,
  "output_tokens": 456,
  "total_tokens": 579,
  "reasoning_time_seconds": 12.345
}
```

注意：全科流式 `done` 当前不包含 `content` 和知识片段字段；前端应通过累积 `token.content` 得到最终回答，并从之前的 `knowledge` 事件保存知识片段。

`error` 事件：

```json
{
  "type": "error",
  "content": "错误信息"
}
```

### 4.4 知识库接口

```http
GET /api/v1/knowledge-base/status
```

响应：

```json
{
  "enabled": true,
  "knowledge_base_dir": "knowledge_base",
  "persist_dir": ".chroma/medical_kb",
  "collection_name": "medical_knowledge",
  "embedding_model": "BAAI/bge-small-zh-v1.5",
  "index_exists": true,
  "indexed_chunks": 86
}
```

```http
POST /api/v1/knowledge-base/rebuild
Content-Type: application/json
```

请求体：

```json
{
  "force_rebuild": true
}
```

响应包含重建消息和最新状态。

前端开发建议：

- 后台管理页可以展示 status。
- 修改知识库 JSON 后，需要调用 rebuild 或让用户在后端执行 `python main.py --rebuild-kb`。
- rebuild 会加载 embedding/reranker 模型，可能耗时较久，前端应显示 loading。

## 5. RAG 与知识库格式

知识库目录由 `MEDICAL_KNOWLEDGE_BASE_DIR` 指定，默认 `knowledge_base/`。程序递归扫描所有 `.json`。

推荐 JSON 结构：

```json
{
  "topic": "血压评估",
  "agent_tags": ["cardiometabolic_health"],
  "assessment_items": [
    {
      "name": "血压分层",
      "description": "..."
    }
  ],
  "rehabilitation": {
    "lifestyle": ["减少钠盐摄入", "规律运动"]
  }
}
```

`agent_tags` 是特殊字段：

- 会在索引时从正文中移除，不会作为知识片段内容注入 prompt。
- 只保留合法专科名称。
- 专科检索会按 `agent_name` 精确过滤。
- 全科检索不传 `agent_name`，会检索全部片段。
- 如果文件没有 `agent_tags` 或全部标签不合法，该文件会以空标签索引；全科可检索，专科过滤时不会命中。

JSON 展开规则：

- 标量会变成 `path: value`。
- 较小结构会整体 flatten 为一个 section。
- 大结构会递归拆分。
- 每个 section 超过 700 字符会切块，overlap 为 120 字符。

检索规则：

1. 用 embedding 模型对查询向量化。
2. Chroma 粗召回 `RAG_COARSE_TOP_K` 条，默认 20。
3. CrossEncoder reranker 重排。
4. 取 `RAG_RERANK_TOP_K` 条作为最终 `knowledge_chunks` 注入 prompt，默认 5。

分数含义：

- `vector_score`: 由 Chroma cosine distance 转换为 `1 - distance` 后截断到非负。
- `rerank_score`: CrossEncoder 输出分数，通常不限制在 0 到 1。
- `score`: 当前最终展示分数；粗召回阶段等于 `vector_score`，重排后等于 `rerank_score`。

## 6. 前端开发对齐建议

当前 Vue 前端位于 `frontend/`，主要页面结构：

- 未登录：品牌化登录/注册页。
- 已登录：三栏工作台。
- 左栏：当前用户、新建健康评估、会话列表。
- 中栏：健康评估对话、模式切换、输入框。
- 右栏：当前模式、路由/token/耗时、知识库依据、安全边界。

UI 定位是健康评估网站，不是通用聊天壳：

- 第一屏和工作台都突出健康评估、生命 8 要素、心理与社会因素。
- 消息角色使用“我的健康信息”“健康评估建议”。
- 侧栏展示知识库依据和路由信息，强调依据和安全边界。

推荐页面能力：

- 文本输入区域。
- 文件上传区域，支持多文件。
- 模式选择：专科路由分析 / 全科综合分析。
- 普通请求和流式请求二选一；正式体验建议优先用流式。
- 主回答区展示最终 `content` 或累积 token。
- 路由结果区展示 `routed_agent_names`、`route_reason`。
- 专科结果区展示每个 `specialist_assessments` 或流式中的 `specialist_result`。
- 知识依据区展示 `knowledge_chunks`，至少显示 `source_file`、`section_path`、`content`。
- 预处理区展示 `source_summary`、`preprocessed_text`、`preprocessing_notes`，建议默认折叠。
- Token 和耗时信息作为调试信息展示。

前端状态机建议：

| 状态 | 触发 |
| --- | --- |
| `idle` | 初始或请求完成 |
| `preprocessing` | 文件流式接口收到请求后，等待 `source` |
| `rewriting` | 收到 `source` 后或文本流式请求开始 |
| `routing` | 收到 `rewrite` |
| `retrieving` | 收到 `route` |
| `specialist_running` | 专科流程收到 `knowledge` 或 `specialist_result` |
| `streaming_answer` | 收到 `token` |
| `done` | 收到 `done` |
| `error` | 收到 `error` 或 HTTP 错误 |

前端兼容注意事项：

- 流式接口不是 OpenAI 风格 SSE，不含 `[DONE]`；以 JSON `{"type":"done"}` 作为结束。
- `token` 只代表最终总结或全科回答的增量，不代表每个专科的增量。
- 专科分析不是 token 级流式输出，而是在每个专科完成后一次性发送 `specialist_result`。
- 专科任务并行执行，`specialist_result` 返回顺序可能不是 router 给出的顺序；普通接口已按 router 顺序排序。
- 文件流式接口如果预处理失败，会直接返回一个 `error` 事件，而不是 HTTP 400 JSON。
- 普通接口错误通常是 FastAPI JSON 错误或 HTTP 500。
- token 统计依赖模型返回 metadata；如果上游不返回，可能为 0。
- 医疗内容要保留安全提示，不要在前端用“诊断结果”这类确定性标题包装。

## 7. 后端开发对齐规则

修改后端时应同步检查：

- 新增或改名 agent 时，同时更新：
  - `ROUTER_AGENT_NAMES`
  - `SPECIALIST_PROMPTS`
  - `SPECIALIST_AGENT_NAMES`
  - fallback router keyword groups
  - 本文档中的 agent 表
  - 知识库 JSON 的 `agent_tags`
- 改响应字段时，同时更新：
  - `schemas.py`
  - 普通接口构造 `build_assessment_response`
  - 流式 `done` 事件
  - 前端解析逻辑
  - 本文档
- 改 RAG 行为时，同时更新：
  - `.env.example`
  - README
  - 本文档 RAG 部分
  - 评测指标解释
- 改文件预处理时，同时更新：
  - 支持格式列表
  - `source_summary` / `preprocessed_text` 行为
  - 前端上传说明

建议保持普通接口和流式接口字段尽量对齐。当前已知不完全对齐点：

- 全科流式 `done` 不返回 `content`、`knowledge_chunks`、`coarse_knowledge_chunks`、`reranked_knowledge_chunks`。
- 全科流式 `knowledge` 不返回 `agent_name` 和 `agent_label`。
- 专科流式的 `specialist_result` 是专科完整结果，不是最终结果。

## 8. 评测与结果文件

批量评测：

```bash
python evaluate_test_data.py --mode specialist --runner direct
python evaluate_test_data.py --mode general --runner direct
```

也可以评测已启动的 API：

```bash
python evaluate_test_data.py --mode specialist --runner api --api-base-url http://127.0.0.1:8000
```

输出目录格式：

```text
eval_results/YYYYMMDD_HHMMSS_{mode}_{runner}/
```

核心文件：

- `summary.json`: 汇总指标。
- `per_case_results.jsonl`: 每个样本的完整结果。
- `manual_review.csv`: 人工或 LLM judge 复核模板。
- `run_config.json`: 本次运行参数。
- `HOW_TO_USE.md`: 评测输出说明。

LLM-as-a-judge：

```bash
python judge_manual_review.py \
  --results-jsonl eval_results/<run>/per_case_results.jsonl \
  --manual-review-csv eval_results/<run>/manual_review.csv \
  --output-csv eval_results/<run>/manual_review_judged.csv \
  --output-jsonl eval_results/<run>/judge_results.jsonl \
  --summary-json eval_results/<run>/judge_summary.json
```

评测数据 `test_data.json` 的关键字段：

| 字段 | 说明 |
| --- | --- |
| `id` | 样本 ID |
| `input_text` | 用户输入 |
| `input_type` | structured / colloquial / incomplete 等 |
| `difficulty` | easy / medium / hard |
| `gold_route` | 期望路由，主要用于专科流程 |
| `gold_relevant_chunks` | 期望命中的知识库文件名 |
| `reference_keypoints` | 人工参考要点 |
| `risk_level` | low / medium / high |
| `safety_expected` | 安全性要求 |
| `notes` | 样本说明 |

自动指标主要用于回归对比，不等同于临床正确性。最终论文或答辩建议结合人工复核或 judge 结果。

## 9. 医疗安全与产品文案边界

这是健康评估和建议系统，不应在 UI 或文档中称为“确诊系统”。

前端文案建议：

- 使用“健康评估”“风险提示”“建议补充信息”“建议线下就医/复查”。
- 避免使用“诊断结果”“治疗处方”“系统判定你患有...”。
- 对高风险内容保留模型输出中的急诊、立即就医、联系身边支持等安全建议。
- 知识库依据不足时，前端应原样展示，不要替用户补全结论。

## 10. 常见开发问题

### 为什么专科接口最后返回 `specialist_summary`？

专科流程可能有多个专科并行参与。最终给用户看的 `content` 是 summary agent 汇总后的结果，因此 `agent_name` 是 `specialist_summary`。实际参与的专科在 `routed_agent_names` 和 `specialist_assessments`。

### 为什么我新增知识库 JSON 后没有生效？

需要重建索引：

```bash
python main.py --rebuild-kb
```

或者调用：

```http
POST /api/v1/knowledge-base/rebuild
```

### 为什么某个专科检索不到某个 JSON？

检查该 JSON 的 `agent_tags` 是否包含对应专科名称。没有标签的文件不会被专科过滤命中，但全科仍可检索。

### 为什么第一次请求很慢？

首次加载 embedding 模型、reranker 模型、Chroma collection 或自动构建索引都会耗时。开发时可先执行 `python main.py --kb-status` 或 `python main.py --rebuild-kb` 预热。

### 为什么 OCR 中文效果差？

图片 OCR 依赖本机 `tesseract` 和 `chi_sim` 中文语言包。缺少 `chi_sim` 时接口会在 `preprocessing_notes` 中提示，中文识别质量可能下降。
