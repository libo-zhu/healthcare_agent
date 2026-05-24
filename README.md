# Healthcare Agent

面向健康评估的多 Agent + RAG 系统。本仓库是本科毕业设计《面向健康评估的智能 Agent 系统开发》的核心工程代码，包含后端 Agent 编排、医学知识库检索、FastAPI 接口、Vue 前端工作台和论文定稿使用的最终实验结果。

## 项目亮点

- 多阶段 Agent 流程：健康信息规范化、专科路由、RAG 检索、四类专科分析、综合总结。
- 本地医学知识库：覆盖血压、血糖、血脂、饮食、BMI、睡眠、运动、戒烟、心理健康与社会决定因素等主题。
- 知识增强生成：使用 Chroma 向量召回和 CrossEncoder 重排序，为专科 Agent 注入相关知识片段。
- 多源输入处理：支持文本、PDF、CSV、TXT 和图片 OCR，适合体检报告和口语化健康咨询。
- 网站平台：FastAPI 后端 + Vue 前端，支持登录、会话管理、文件上传、流式响应和知识依据展示。
- 实验评估：使用 80 条健康咨询样本，与 DeepSeek V4 直接调用基线对比，并采用 GPT-5.5 high 进行 KB-aware 成对评审。

## 最终实验结果

以下结果以 `../thesis/本科毕设论文-朱力波.docx` 定稿第 4 章为准。

| 指标 | 结果 |
| --- | ---: |
| 测试样本数 | 80 |
| 路由准确率 | 0.8729 |
| 路由召回率 | 0.9875 |
| 路由 F1 值 | 0.9058 |
| RAG 检索准确率 | 0.4860 |
| RAG 检索召回率 | 0.8152 |
| RAG 检索 F1 值 | 0.5805 |
| 高风险样本紧急提示覆盖率 | 100% |
| 高风险样本延误提示率 | 0% |
| Agent 更优 / DeepSeek 更优 / 二者相当 | 35 / 3 / 42 |
| Agent 不劣于基线比例 | 96.25% |

四维评分结果：

| 评价维度 | Healthcare Agent | DeepSeek V4 | 差值 |
| --- | ---: | ---: | ---: |
| 正确性 | 4.3000 | 4.4875 | -0.1875 |
| 覆盖度 | 4.9500 | 4.5375 | +0.4125 |
| 依据充分性 | 4.6875 | 4.0125 | +0.6750 |
| 安全性 | 4.8750 | 4.8500 | +0.0250 |

论文定稿对应的实验文件：

```text
eval_results/20260428_185414_specialist_direct/      # Healthcare Agent 最终运行结果
eval_results/20260428_190657_specialist_deepseek/    # DeepSeek V4 直接调用基线
eval_results/gpt55_high_kb_aware_judge/              # GPT-5.5 high KB-aware 成对评审
eval_results/EXPERIMENT_ANALYSIS.md                  # 实验分析汇总
```

## 项目结构

```text
src/healthcare_agent/
  agent.py            # Agent 编排、流式事件、token 聚合
  api.py              # FastAPI 接口
  auth.py             # 登录 token、密码哈希、当前用户
  chat_service.py     # 会话、消息、上下文评估
  cli.py              # 命令行入口
  config.py           # 环境变量与默认配置
  database.py         # MySQL 连接与表结构初始化
  knowledge_base.py   # JSON 知识库索引、检索、重排
  preprocessing.py    # 文件解析、PDF 提取、图片 OCR
  prompts.py          # 规范化、路由、专科、全科、总结 prompt
  schemas.py          # 请求响应 schema

frontend/             # Vue 健康评估工作台
knowledge_base/       # 本地医学 JSON 知识库
eval_results/         # 最终实验结果与分析
docs/                 # 开发对齐文档和项目周志
evaluate_test_data.py # 批量评测脚本
judge_manual_review.py
judge_pairwise_comparison.py
main.py               # CLI 启动入口
run_api.py            # FastAPI 启动入口
test_data.json        # 80 条实验测试样本
```

## 快速开始

要求 Python `>=3.12`。

```bash
pip install -e .
cp .env.example .env
```

在 `.env` 中配置 DeepSeek、MySQL 和 RAG 参数：

```env
DEEPSEEK_API_KEY=your_api_key
AUTH_SECRET_KEY=replace_with_a_random_local_secret
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
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

构建或重建知识库索引：

```bash
python main.py --rebuild-kb
```

命令行试用：

```bash
python main.py
```

启动后端 API：

```bash
python run_api.py
```

默认接口地址为 `http://127.0.0.1:8000`，Swagger 文档为 `http://127.0.0.1:8000/docs`。

启动前端：

```bash
cd frontend
npm install
npm run dev
```

前端默认地址为 `http://localhost:5173/`。

## 核心接口

| 功能 | 接口 |
| --- | --- |
| 专科健康评估 | `POST /api/v1/specialist/assessment` |
| 全科综合评估 | `POST /api/v1/general/assessment` |
| 专科流式评估 | `POST /api/v1/specialist/assessment/stream` |
| 全科流式评估 | `POST /api/v1/general/assessment/stream` |
| 文件上传专科评估 | `POST /api/v1/specialist/assessment/files` |
| 文件上传全科评估 | `POST /api/v1/general/assessment/files` |
| 知识库状态 | `GET /api/v1/knowledge-base/status` |
| 重建知识库 | `POST /api/v1/knowledge-base/rebuild` |
| 注册 / 登录 | `POST /api/v1/auth/register`、`POST /api/v1/auth/login` |
| 会话列表 | `GET /api/v1/conversations` |
| 会话消息 | `POST /api/v1/conversations/{conversation_id}/messages` |

请求示例：

```json
{
  "medical_data": "患者，45岁，男性，血压148/95 mmHg，空腹血糖6.8 mmol/L，最近经常熬夜，偶尔头晕。"
}
```

更完整的响应字段、SSE 事件格式、文件上传规则和前后端对齐说明见 [docs/DEVELOPMENT_ALIGNMENT.md](docs/DEVELOPMENT_ALIGNMENT.md)。

## 复现实验

运行 Healthcare Agent 最终链路：

```bash
python evaluate_test_data.py --mode specialist --runner direct
```

运行 DeepSeek V4 直接调用基线：

```bash
python evaluate_test_data.py --mode specialist --runner deepseek
```

评测结果会写入：

```text
eval_results/YYYYMMDD_HHMMSS_{mode}_{runner}/
```

本仓库保留的最终论文结果对应：

```text
eval_results/20260428_185414_specialist_direct/
eval_results/20260428_190657_specialist_deepseek/
eval_results/gpt55_high_kb_aware_judge/
```

## 知识库格式

知识库默认放在 `knowledge_base/`，程序递归扫描 `.json` 文件。推荐结构：

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

`agent_tags` 用于专科检索过滤，合法值包括：

- `sleep_activity_nicotine`
- `diet_bmi`
- `cardiometabolic_health`
- `mental_social_health`

修改知识库后需要重建索引：

```bash
python main.py --rebuild-kb
```

## 安全边界

本系统定位为健康评估和健康管理建议辅助，不是确诊系统或处方系统。输出应使用“风险提示”“建议复查”“建议就医”等表达；涉及急症、自伤风险、严重高血压或严重高血糖等高风险情况时，应优先提示及时线下就医或紧急求助。
