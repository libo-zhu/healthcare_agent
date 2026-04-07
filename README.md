# healthcare_agent

毕业设计，面向健康评估的智能 agent 开发，agent 部分的代码。

## 当前示例

这个仓库现在包含一个基于 LangChain + DeepSeek 的最小健康评估 agent：

- 按文件拆分了配置、prompt、agent 执行和命令行入口
- 默认使用 `deepseek-chat`
- 只做 prompt engineering，不接工具、不接数据库
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

## 安全说明

`.gitignore` 已忽略以下内容：

- `.venv/`
- `.env`
- `.env.*`（但保留 `.env.example`）

这样你的 DeepSeek API key 默认不会被提交到 GitHub。
