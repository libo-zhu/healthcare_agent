# healthcare_agent

毕业设计，面向健康评估的智能 agent 开发，agent 部分的代码。

## 当前示例

这个仓库现在包含一个基于 LangChain + DeepSeek 的最小健康评估 agent：

- 按文件拆分了配置、prompt、agent 执行和命令行入口
- 默认使用 `deepseek-chat`
- 采用“1 个路由 agent + 4 个专科 agent”的 prompt engineering 结构
- 直接打印模型输出，用于验证 API 是否正常

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

3. 运行

```bash
python main.py
```

程序会提示输入医疗数据；如果你直接回车，会使用内置测试样例。

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

### 普通接口

`POST /api/v1/health-assessment`

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
  "preprocessed_text": "患者，45岁，男性，血压148/95 mmHg，空腹血糖6.8 mmol/L，最近经常熬夜，偶尔头晕。"
}
```

### 流式接口

`POST /api/v1/health-assessment/stream`

返回类型为 `text/event-stream`，每一段数据格式类似：

```text
data: {"type":"route","agent_name":"mental_social_health","agent_label":"Mental Health / SDOH","reason":"..."}
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

接口如下：

- `POST /api/v1/health-assessment/files`
- `POST /api/v1/health-assessment/files/stream`

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

### 1. 测普通文本接口

请求：

- 方法：`POST`
- URL：`http://127.0.0.1:8000/api/v1/health-assessment`
- Body 选择 `raw`
- 格式选择 `JSON`

示例请求体：

```json
{
  "medical_data": "患者，45岁，男性，血压148/95 mmHg，空腹血糖6.8 mmol/L，最近经常熬夜，偶尔头晕。"
}
```

你会看到模型输出以及 token 统计，还有 `preprocessed_text`。

### 2. 测文本流式接口

请求：

- 方法：`POST`
- URL：`http://127.0.0.1:8000/api/v1/health-assessment/stream`
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

### 3. 测文件上传接口

请求：

- 方法：`POST`
- URL：`http://127.0.0.1:8000/api/v1/health-assessment/files`
- Body 选择 `form-data`

字段：

- `medical_data`：可选，类型选 `Text`
- `files`：可选，类型选 `File`

你可以上传一个或多个 `txt/csv/pdf/图片` 文件。若同时传 `medical_data`，系统会把文本说明和提取出的文件内容合并后再分析。

### 4. 测文件流式接口

请求：

- 方法：`POST`
- URL：`http://127.0.0.1:8000/api/v1/health-assessment/files/stream`
- Body 选择 `form-data`

字段同上。

返回顺序通常是：

```text
data: {"type":"source","source_summary":"inline_text, report.txt","preprocessed_text":"..."}
data: {"type":"token","content":"..."}
data: {"type":"token","content":"..."}
data: {"type":"done","input_tokens":123,"output_tokens":456,"total_tokens":579}
```

### 5. 常见问题

- 上传图片时报 `tesseract` 相关错误：说明本机还没安装系统级 OCR 依赖，`txt/csv/pdf` 仍可正常使用。
- 流式接口在 Postman 中如果显示不明显：可以先用普通接口确认结果，再观察 Postman 的实时响应输出区域。

## 安全说明

`.gitignore` 已忽略以下内容：

- `.venv/`
- `.env`
- `.env.*`（但保留 `.env.example`）

这样你的 DeepSeek API key 默认不会被提交到 GitHub。
