# healthcare_agent

毕业设计，面向健康评估的智能 agent 开发，agent 部分的代码。

## 当前示例

这个仓库现在包含一个基于 LangChain + DeepSeek 的最小健康评估 agent：

- 按文件拆分了配置、prompt、agent 执行和命令行入口
- 默认使用 `deepseek-chat`
- 采用“1 个路由 agent + 4 个专科 agent”的 prompt engineering 结构
- 额外提供 1 个全科综合分析 agent
- 新增基于本地 JSON 医学知识库的 RAG 检索能力
- 检索结果会作为知识依据注入 prompt，并在接口响应中返回

## RAG 设计

当前实现采用：

- 向量数据库：`Chroma`，本地持久化，无需单独部署服务
- 向量模型：默认 `BAAI/bge-small-zh-v1.5`
- 知识源：你本地的 `.json` 文件
- 检索流程：`JSON -> 结构展开 -> 文本分块 -> 向量化 -> Chroma 检索 -> 把命中的知识片段拼进 prompt`

为了尽量满足“严格依据知识库回答”，专科 agent 和全科 agent 的 prompt 已增加约束：

- 只能依据“用户输入 + 检索到的知识片段”回答
- 如果知识库依据不足，必须明确写出“知识库依据不足”
- 不允许自由补充知识库外的医学结论

## 你的 JSON 怎么使用

请把现有 20 个医学 JSON 文件放到项目根目录的 `knowledge_base/` 下，支持多级子目录，例如：

```text
knowledge_base/
  life8/
    sleep.json
    activity.json
    nicotine.json
  rehab/
    hypertension_rehab.json
    diabetes_rehab.json
```

程序会递归扫描 `knowledge_base/**/*.json`。

建议你的 JSON 内容尽量保持清晰的层级结构，例如：

```json
{
  "topic": "血压评估",
  "agent_tags": ["cardiometabolic_health"],
  "assessment_items": [
    {
      "name": "血压分层",
      "description": "......"
    }
  ],
  "rehabilitation": {
    "lifestyle": [
      "减少钠盐摄入",
      "规律运动"
    ]
  }
}
```

其中 `agent_tags` 用于控制专科 agent 的检索范围。当前支持：

- `sleep_activity_nicotine`
- `diet_bmi`
- `cardiometabolic_health`
- `mental_social_health`

专科 agent 会优先只检索带有自己标签的知识片段；全科 agent 仍会检索全部知识库内容。

当前实现会自动把 JSON 展开成带路径的文本片段，例如：

```text
assessment_items[0].name: 血压分层
assessment_items[0].description: ......
rehabilitation.lifestyle[0]: 减少钠盐摄入
```

然后再做切块和向量检索，所以不需要你手工先转成 txt。

## 运行方式

1. 激活虚拟环境

```bash
source .venv/bin/activate
```

2. 在项目根目录创建本地 `.env`

```bash
cp .env.example .env
```

然后把 `.env` 里的 `DEEPSEEK_API_KEY` 改成你的真实 key。

如果你要启用 RAG，也建议同步检查这些配置：

```env
RAG_ENABLED=true
RAG_AUTO_BUILD=true
RAG_TOP_K=4
MEDICAL_KNOWLEDGE_BASE_DIR=knowledge_base
CHROMA_PERSIST_DIR=.chroma/medical_kb
VECTOR_COLLECTION_NAME=medical_knowledge
EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5
```

其中：

- `MEDICAL_KNOWLEDGE_BASE_DIR`：你的 JSON 知识库目录
- `CHROMA_PERSIST_DIR`：本地向量索引存储目录
- `EMBEDDING_MODEL_NAME`：可写 HuggingFace 模型名，也可以替换成本地模型目录

3. 安装新增依赖

```bash
pip install -e .
```

如果你之前已经装过旧版本依赖，请重新执行一次，确保安装 `chromadb` 和 `sentence-transformers`。

4. 先构建知识库索引

```bash
python main.py --rebuild-kb
```

如果一切正常，你会看到索引了多少个 `json` 文件、多少个知识片段。修改 JSON 内容或 `agent_tags` 后，也需要重新执行一次索引重建。

5. 查看知识库状态

```bash
python main.py --kb-status
```

6. 运行

```bash
python main.py
```

程序会提示输入医疗数据；如果你直接回车，会使用内置测试样例。

首次调用时，如果本地还没有索引，并且 `RAG_AUTO_BUILD=true`，程序会自动尝试从 `knowledge_base/` 构建索引。

## 为什么选 Chroma

你这套场景里知识库规模不大，约 20 个 JSON 文件，本地开发阶段优先推荐 Chroma：

- 集成简单，直接嵌入 Python 项目
- 支持持久化到本地目录
- 不需要单独起 Milvus / Elasticsearch / Qdrant 服务
- 后续如果数据量变大，再迁移到 Qdrant / Milvus 也比较自然

如果后续你需要：

- 更大规模数据
- 多人共享向量库
- 更强的过滤 / 运维能力

那时再迁移到 `Qdrant` 会更合适。

## FastAPI 接口

安装完成后可以直接启动服务：

```bash
source .venv/bin/activate
python run_api.py
```

服务默认地址：

```text
http://127.0.0.1:8000
```

Swagger 文档：

```text
http://127.0.0.1:8000/docs
```

### CORS

服务已开启开发阶段可直接联调的 CORS 配置：

- 允许任意来源 `*`
- 允许任意请求头
- 允许任意方法

这意味着你本地前端页面可以直接请求这个 FastAPI 服务。

## 接口分组

当前共有 8 个核心接口：

- 专科路由分析：`POST /api/v1/specialist/assessment`
- 专科路由分析流式：`POST /api/v1/specialist/assessment/stream`
- 专科路由分析文件上传：`POST /api/v1/specialist/assessment/files`
- 专科路由分析文件上传流式：`POST /api/v1/specialist/assessment/files/stream`
- 全科综合分析：`POST /api/v1/general/assessment`
- 全科综合分析流式：`POST /api/v1/general/assessment/stream`
- 全科综合分析文件上传：`POST /api/v1/general/assessment/files`
- 全科综合分析文件上传流式：`POST /api/v1/general/assessment/files/stream`
- 知识库状态：`GET /api/v1/knowledge-base/status`
- 重建知识库索引：`POST /api/v1/knowledge-base/rebuild`

其中：

- `specialist` 组会先经过路由 agent，再交给 4 个专科 agent 之一
- `general` 组会直接交给全科综合分析 agent，不做专科路由

### 专科普通接口

`POST /api/v1/specialist/assessment`

请求体示例：

```json
{
  "medical_data": "患者，45岁，男性，血压148/95 mmHg，空腹血糖6.8 mmol/L，最近经常熬夜，偶尔头晕。"
}
```

响应示例：

```json
{
  "content": "......",
  "agent_name": "cardiometabolic_health",
  "input_tokens": 123,
  "output_tokens": 456,
  "total_tokens": 579,
  "source_summary": "inline_text",
  "preprocessed_text": "患者，45岁，男性，血压148/95 mmHg，空腹血糖6.8 mmol/L，最近经常熬夜，偶尔头晕。",
  "knowledge_chunks": [
    {
      "source_file": "life8/blood_pressure.json",
      "section_path": "root.assessment_items[0]",
      "content": "...",
      "score": 0.87
    }
  ]
}
```

`knowledge_chunks` 就是本次回答实际检索命中的知识依据。

### 专科流式接口

`POST /api/v1/specialist/assessment/stream`

返回类型为 `text/event-stream`，每一段数据格式类似：

```text
data: {"type":"route","agent_name":"mental_social_health","agent_label":"Mental Health / SDOH","reason":"..."}
data: {"type":"knowledge","chunks":[{"source_file":"...","section_path":"...","content":"...","score":0.91}]}
data: {"type":"token","content":"..."}
```

结束时会返回：

```text
data: {"type":"done","agent_name":"mental_social_health","input_tokens":123,"output_tokens":456,"total_tokens":579}
```

这条接口适合前端流式渲染，也可以在 Postman 中观察分块输出。

## 文件上传接口

除了直接传文本，现在还支持上传文件并先做文本预处理，再交给大模型分析。

支持格式：

- `.txt`
- `.csv`
- `.pdf`
- 图片 OCR：`.png`、`.jpg`、`.jpeg`、`.bmp`、`.tif`、`.tiff`、`.webp`

专科路由接口：

- `POST /api/v1/specialist/assessment/files`
- `POST /api/v1/specialist/assessment/files/stream`

全科综合接口：

- `POST /api/v1/general/assessment/files`
- `POST /api/v1/general/assessment/files/stream`

这两个接口使用 `multipart/form-data`：

- `medical_data`: 可选，补充说明文字
- `files`: 可选，可上传一个或多个文件

你至少需要传 `medical_data` 或 `files` 其中之一。

普通上传接口返回里会额外包含：

- `source_summary`: 本次输入包含哪些来源
- `preprocessed_text`: 上传文件经过提取/OCR/拼接后，最终送给大模型的文本

流式上传接口开始时会先返回一个 `source` 事件，方便你调试预处理结果：

```text
data: {"type":"source","source_summary":"inline_text, report.pdf","preprocessed_text":"..."}
```

### OCR 说明

图片 OCR 目前通过本地 `tesseract` 完成，适合识别体检单、化验单、报告截图这类以文字为主的图片。

如果你本机还没有安装 `tesseract`，图片 OCR 接口会报错提示；`txt/csv/pdf` 不受影响。

## Postman 测试说明

### 1. 测专科普通文本接口

请求：

- 方法：`POST`
- URL：`http://127.0.0.1:8000/api/v1/specialist/assessment`
- Body 选择 `raw`
- 格式选择 `JSON`

示例请求体：

```json
{
  "medical_data": "患者，45岁，男性，血压148/95 mmHg，空腹血糖6.8 mmol/L，最近经常熬夜，偶尔头晕。"
}
```

你会看到模型输出以及 token 统计，还有 `preprocessed_text`。

### 2. 测专科文本流式接口

请求：

- 方法：`POST`
- URL：`http://127.0.0.1:8000/api/v1/specialist/assessment/stream`
- Body 选择 `raw`
- 格式选择 `JSON`

示例请求体：

```json
{
  "medical_data": "患者，52岁，女性，最近体检提示总胆固醇偏高，近两周胸闷，睡眠一般。"
}
```

返回会持续追加多条 SSE 数据：

```text
data: {"type":"token","content":"..."}
data: {"type":"token","content":"..."}
data: {"type":"done","input_tokens":123,"output_tokens":456,"total_tokens":579}
```

### 3. 测专科文件上传接口

请求：

- 方法：`POST`
- URL：`http://127.0.0.1:8000/api/v1/specialist/assessment/files`
- Body 选择 `form-data`

字段：

- `medical_data`：可选，类型选 `Text`
- `files`：可选，类型选 `File`

你可以上传一个或多个 `txt/csv/pdf/图片` 文件。若同时传 `medical_data`，系统会把文本说明和提取出的文件内容合并后再分析。

### 4. 测专科文件流式接口

请求：

- 方法：`POST`
- URL：`http://127.0.0.1:8000/api/v1/specialist/assessment/files/stream`
- Body 选择 `form-data`

字段同上。

返回顺序通常是：

```text
data: {"type":"source","source_summary":"inline_text, report.txt","preprocessed_text":"..."}
data: {"type":"knowledge","chunks":[...]}
data: {"type":"token","content":"..."}
data: {"type":"token","content":"..."}
data: {"type":"done","input_tokens":123,"output_tokens":456,"total_tokens":579}
```

## 知识库接口

### 查看状态

`GET /api/v1/knowledge-base/status`

返回示例：

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

### 重建索引

`POST /api/v1/knowledge-base/rebuild`

请求体：

```json
{
  "force_rebuild": true
}
```

适合你更新了 JSON 文件之后主动重建索引。

## 使用建议

1. 先把生命 8 要素和康复干预内容按主题拆分成多个 JSON 文件，尽量让每个文件主题单一。
2. 每条指标说明、风险解释、干预建议尽量用结构化字段表达，不要把所有内容塞进一个超长字符串。
3. 如果你的知识库特别强调“严谨”，建议在 JSON 中加入 `source`、`version`、`updated_at` 这类字段，后续可以直接展示来源。
4. 如果你后面希望“按专科只检索对应知识子库”，可以再给 JSON 增加 `category` 字段，我可以下一步帮你做 metadata 过滤。

## 依赖安装常见问题

### 1. `sentence-transformers` 下载模型慢或失败

默认会从 HuggingFace 下载嵌入模型。如果你网络环境不方便：

- 可以提前手工下载模型到本地
- 然后把 `EMBEDDING_MODEL_NAME` 改成本地目录

例如：

```env
EMBEDDING_MODEL_NAME=/absolute/path/to/bge-small-zh-v1.5
```

### 2. 不想现在就上更重的向量数据库

目前没必要一开始就装 Milvus、Qdrant、ES。对于你这个毕业设计阶段的本地知识库，Chroma 已经足够。

### 5. 测全科普通文本接口

请求：

- 方法：`POST`
- URL：`http://127.0.0.1:8000/api/v1/general/assessment`
- Body 选择 `raw`
- 格式选择 `JSON`

这组接口不会经过专科路由，而是直接由全科综合分析 agent 输出整体建议。

### 6. 测全科流式接口

请求：

- 方法：`POST`
- URL：`http://127.0.0.1:8000/api/v1/general/assessment/stream`
- Body 选择 `raw`
- 格式选择 `JSON`

### 7. 测全科文件上传接口

请求：

- 方法：`POST`
- URL：`http://127.0.0.1:8000/api/v1/general/assessment/files`
- Body 选择 `form-data`

### 8. 测全科文件流式接口

请求：

- 方法：`POST`
- URL：`http://127.0.0.1:8000/api/v1/general/assessment/files/stream`
- Body 选择 `form-data`

### 5. 常见问题

- 上传图片时报 `tesseract` 相关错误：说明本机还没安装系统级 OCR 依赖，`txt/csv/pdf` 仍可正常使用。
- 流式接口在 Postman 中如果显示不明显：可以先用普通接口确认结果，再观察 Postman 的实时响应输出区域。

## 安全说明

`.gitignore` 已忽略以下内容：

- `.venv/`
- `.env`
- `.env.*`（但保留 `.env.example`）

这样你的 DeepSeek API key 默认不会被提交到 GitHub。
