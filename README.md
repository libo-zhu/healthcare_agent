# healthcare_agent

毕业设计项目：面向健康评估的智能 agent 系统。当前仓库包含 agent 编排、FastAPI 接口、本地 JSON 知识库 RAG、文件上传预处理、MySQL 用户与会话持久化、Vue 前端、批量评测脚本。

面向后续后端与前端网页开发的完整对齐文档见：

- [docs/DEVELOPMENT_ALIGNMENT.md](docs/DEVELOPMENT_ALIGNMENT.md)

后续如果接口、agent 流程、知识库格式或前端解析方式发生变化，请同步更新这份文档，避免前后端信息不一致。

## 当前能力

- 基于 LangChain + DeepSeek 的健康评估 agent。
- `1 个 rewrite agent + 1 个 router agent + 4 个专科 agent + 1 个专科总结 agent` 的专科流程。
- 额外提供 `general_health_overview` 全科综合分析流程。
- 基于本地 JSON 知识库的 RAG：Chroma 粗召回 + CrossEncoder reranker 重排。
- 支持普通 JSON 接口和 SSE 流式接口。
- 支持注册、登录、新建会话、对话历史持久化和轻量上下文记忆。
- 提供 Vue 健康评估工作台，包含会话列表、模式切换、知识依据侧栏。
- 支持上传 `.txt`、`.csv`、`.pdf` 和常见图片格式；图片 OCR 依赖本机 `tesseract`。
- 提供自动评测与 LLM-as-a-judge 辅助评分脚本。

## 项目结构

```text
src/healthcare_agent/
  agent.py            # agent 编排、流式事件、token 聚合
  api.py              # FastAPI 接口
  cli.py              # 命令行入口
  config.py           # 环境变量与默认配置
  knowledge_base.py   # JSON 知识库索引、检索、重排
  preprocessing.py    # 文件解析、PDF 提取、图片 OCR
  prompts.py          # router/rewrite/专科/全科/总结 prompt
  schemas.py          # 请求响应 schema
  database.py         # MySQL 连接与表结构初始化
  auth.py             # 登录 token、密码哈希、当前用户
  chat_service.py     # 会话、消息、上下文评估

frontend/             # Vue 健康评估网页
knowledge_base/       # 本地医学 JSON 知识库
evaluate_test_data.py # 批量评测
judge_manual_review.py# LLM-as-a-judge 评分
main.py               # CLI 启动入口
run_api.py            # FastAPI 启动入口
test_data.json        # 评测数据
```

## 快速开始

要求 Python `>=3.12`。

```bash
pip install -e .
cp .env.example .env
```

把 `.env` 里的 `DEEPSEEK_API_KEY` 改成真实 key。

常用 RAG 配置：

```env
AUTH_SECRET_KEY=replace_with_a_random_local_secret
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=zhu203926
MYSQL_DATABASE=healthcare_agent_app
RAG_ENABLED=true
RAG_AUTO_BUILD=true
RAG_TOP_K=5
RAG_COARSE_TOP_K=20
RAG_RERANK_TOP_K=5
MEDICAL_KNOWLEDGE_BASE_DIR=knowledge_base
CHROMA_PERSIST_DIR=.chroma/medical_kb
VECTOR_COLLECTION_NAME=medical_knowledge
EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5
RERANKER_MODEL_NAME=BAAI/bge-reranker-base
```

构建知识库索引：

```bash
python main.py --rebuild-kb
```

查看知识库状态：

```bash
python main.py --kb-status
```

命令行试用：

```bash
python main.py
```

启动 API：

```bash
python run_api.py
```

默认地址：

```text
http://127.0.0.1:8000
```

Swagger：

```text
http://127.0.0.1:8000/docs
```

启动前端：

```bash
cd frontend
npm install
npm run dev
```

前端地址：

```text
http://localhost:5173/
```

## 核心接口

文本评估：

- `POST /api/v1/specialist/assessment`
- `POST /api/v1/general/assessment`

文本流式评估：

- `POST /api/v1/specialist/assessment/stream`
- `POST /api/v1/general/assessment/stream`

文件上传评估：

- `POST /api/v1/specialist/assessment/files`
- `POST /api/v1/general/assessment/files`

文件上传流式评估：

- `POST /api/v1/specialist/assessment/files/stream`
- `POST /api/v1/general/assessment/files/stream`

知识库：

- `GET /api/v1/knowledge-base/status`
- `POST /api/v1/knowledge-base/rebuild`

登录与会话：

- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`
- `GET /api/v1/conversations`
- `POST /api/v1/conversations`
- `GET /api/v1/conversations/{conversation_id}`
- `PATCH /api/v1/conversations/{conversation_id}`
- `DELETE /api/v1/conversations/{conversation_id}`
- `POST /api/v1/conversations/{conversation_id}/messages`

请求示例：

```json
{
  "medical_data": "患者，45岁，男性，血压148/95 mmHg，空腹血糖6.8 mmol/L，最近经常熬夜，偶尔头晕。"
}
```

完整响应字段、SSE 事件格式、文件上传规则和前端解析注意事项见 [开发对齐文档](docs/DEVELOPMENT_ALIGNMENT.md)。

## 知识库 JSON

知识库默认放在 `knowledge_base/`，程序会递归扫描所有 `.json` 文件。

推荐结构：

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

`agent_tags` 用于专科检索过滤。当前合法值：

- `sleep_activity_nicotine`
- `diet_bmi`
- `cardiometabolic_health`
- `mental_social_health`

修改 JSON 后请重建索引：

```bash
python main.py --rebuild-kb
```

## 评测

直接调用本地 Python agent：

```bash
python evaluate_test_data.py --mode specialist --runner direct
python evaluate_test_data.py --mode general --runner direct
```

调用已启动的 API：

```bash
python evaluate_test_data.py --mode specialist --runner api --api-base-url http://127.0.0.1:8000
```

评测结果会写入：

```text
eval_results/YYYYMMDD_HHMMSS_{mode}_{runner}/
```

LLM-as-a-judge 示例：

```bash
python judge_manual_review.py \
  --results-jsonl eval_results/<run>/per_case_results.jsonl \
  --manual-review-csv eval_results/<run>/manual_review.csv \
  --output-csv eval_results/<run>/manual_review_judged.csv \
  --output-jsonl eval_results/<run>/judge_results.jsonl \
  --summary-json eval_results/<run>/judge_summary.json
```

## 安全边界

这个系统定位为健康评估和建议辅助，不是确诊系统或处方系统。前端和文档中应使用“健康评估”“风险提示”“建议复查/就医”等表达，避免把输出包装成确定性诊断。

`.gitignore` 已忽略 `.venv/`、`.env`、`.env.*`，保留 `.env.example`。
