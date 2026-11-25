# Agentic AI Lab（Track B 实战练习仓）

基于 [WenyuChiou/awesome-agentic-ai-zh](https://github.com/WenyuChiou/awesome-agentic-ai-zh) 的 Track B（Agent Builder）脱框架手写练习项目。

## 当前进度：Stage 3 — Tool Use & 第一个 Agent Loop

目录：`stage03_tool_use/`

特点：
- **零第三方依赖**：仅用 Python 标准库 `urllib.request`，直接暴露 OpenAI 兼容协议的原始 JSON 请求与响应，彻底拆开 LangChain4j / LangChain 的黑盒。
- **内置 `--mock` 模式**：不填 API Key 也能先跑通完整的多步 Tool Call + 并行调用 + 错误自愈流程，看清 `messages` 数组的每一步演变。
- **内置 3 个本地工具**：`list_notes`、`read_note`、`calculate`（基于 AST 的安全计算器）。

## 快速开始

### 1. 先用 Mock 模式看清完整协议（免 Key）

```bash
python3 stage03_tool_use/hello_agent.py --mock --dump-messages
```

### 2. 接入你的 OpenAI 兼容 API 实战

```bash
cp .env.example .env
# 编辑 .env，填入你的 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
python3 stage03_tool_use/hello_agent.py
```

自定义提问：

```bash
python3 stage03_tool_use/hello_agent.py --query "帮我查一下笔记里 Q3 预算和服务器月度费用，算算一季度（3个月）总开销占预算的百分比"
```

## Stage 3 三个动手小实验（推荐按顺序做完）

1. **实验 A｜看清底层协议**：加 `--dump-messages` 跑一次，观察每一轮 `assistant.tool_calls[].id` 和 `tool.tool_call_id` 是如何一一配对的。
2. **实验 B｜触发工具报错自愈**：故意问一个不存在的文件（例如 `"帮我读一下 budget.md 里的 Q3 预算"`），观察 `read_note` 返回 `{"ok": false, "error": ...}` 后，模型如何在下一轮主动改调 `list_notes` 修正错误。
3. **实验 C｜触发 `max_steps` 护栏 + 自己加第 4 个工具**：
   - 把 `--max-steps 2` 传进去，确认 Agent 在超步数时安全退出而不是死循环；
   - 在 `hello_agent.py` 里新增一个工具（例如 `search_notes(keyword)`），同时补齐它的 JSON Schema 和分发逻辑，体会 `Tool Schema` 写得清不清楚如何直接影响模型调用稳定性。
