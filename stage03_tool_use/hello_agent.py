#!/usr/bin/env python3
"""
Stage 3：脱框架手写第一个 Agent Loop（零外部依赖，兼容所有 OpenAI 格式接口）

核心八词契约对照：
1. Tool Schema      -> TOOLS 里的 JSON Schema（给模型看的工具说明书）
2. Tool Use         -> 请求体中携带 tools，允许模型决定是否调用工具
3. Function Calling -> 模型返回结构化函数名 + 参数，而非自己执行代码
4. Tool Call        -> assistant 消息里的 tool_calls 数组（含 id / name / arguments）
5. Tool Result      -> role="tool" 且带对应 tool_call_id 的回填消息
6. Agent Loop       -> while step < max_steps 的推理-执行-观察循环
7. ReAct            -> 思考 -> 调工具行动 -> 观察工具结果 -> 决定继续或回答
8. Structured Output-> 工具参数严格遵循 JSON Schema + 代码层做业务校验
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

BASE_DIR = Path(__file__).resolve().parent
NOTES_DIR = BASE_DIR / "sample_notes"
REPO_ROOT = BASE_DIR.parent

SYSTEM_PROMPT = """你是一个严谨的本地助手，必须基于工具返回的真实数据回答问题，不要编造数字或心算复杂算式。

规则：
1. 不知道有哪些笔记时，先调用 list_notes 查看文件名。
2. 需要计算金额、汇总、百分比时，务必调用 calculate 工具，不要口算。
3. 如果工具返回错误（ok=false），根据错误信息调整参数重试；无法解决时如实说明。
4. 最终回答需列出数据来源文件名和计算过程。"""


# =============================================================================
# 1. 本地工具实现（真实执行发生在你自己的进程里，而不是 LLM 里）
# =============================================================================

def tool_list_notes() -> dict[str, Any]:
    files = sorted(p.name for p in NOTES_DIR.glob("*.md"))
    return {"files": files, "count": len(files)}


def tool_read_note(filename: str) -> dict[str, Any]:
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError("filename 必须是纯文件名，不能包含路径分隔符")
    target = NOTES_DIR / filename
    if not target.exists():
        available = sorted(p.name for p in NOTES_DIR.glob("*.md"))
        raise FileNotFoundError(
            f"找不到文件 '{filename}'。可用文件: {available}"
        )
    return {"filename": filename, "content": target.read_text(encoding="utf-8")}


# 安全四则运算器（用 AST 白名单，绝不用危险的 eval）
_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval_node(node.operand))
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


def tool_calculate(expression: str) -> dict[str, Any]:
    tree = ast.parse(expression.strip(), mode="eval")
    value = _safe_eval_node(tree)
    return {"expression": expression, "result": round(value, 4)}


TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "list_notes": tool_list_notes,
    "read_note": tool_read_note,
    "calculate": tool_calculate,
}


# =============================================================================
# 2. Tool Schema（给 LLM 看的「工具说明书」，对应 LangChain4j 的 @Tool + @P）
# =============================================================================

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": "列出本地 sample_notes 目录下所有可读的 Markdown 笔记文件名。",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_note",
            "description": "按文件名读取一篇笔记的完整文本内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "笔记文件名，例如 q3_budget.md（不要带目录路径）",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "精确计算数学四则运算表达式，支持加减乘除与括号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "标准数学表达式，例如 '(8200 + 2400 + 1800 + 2600) * 3 / 48000 * 100'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


# =============================================================================
# 3. 工具分发器：校验参数 + 捕获异常回传（工程稳定性的核心护栏）
# =============================================================================

def execute_tool(name: str, raw_arguments: str) -> str:
    """永远返回可解析的 JSON 字符串；出错时不崩进程，而是把清晰错误喂回给模型。"""
    func = TOOL_REGISTRY.get(name)
    if func is None:
        return json.dumps(
            {"ok": False, "error": f"未知工具 '{name}'，可用工具: {list(TOOL_REGISTRY)}"},
            ensure_ascii=False,
        )

    try:
        args = json.loads(raw_arguments) if raw_arguments else {}
        if not isinstance(args, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        data = func(**args)
        return json.dumps({"ok": True, "data": data}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            ensure_ascii=False,
        )


# =============================================================================
# 4. 原始 HTTP 调用 OpenAI 兼容接口（或 --mock 离线模拟）
# =============================================================================

def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def call_openai_compatible(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "")

    if not base_url or not api_key or not model:
        raise RuntimeError(
            "缺少配置：请先复制 .env.example 为 .env，并填写 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL"
            "（或者加 --mock 参数离线体验完整协议）。"
        )

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,  # 稳定输出第一原则：工具调用场景默认 temperature=0
    }
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait_sec = 2 ** attempt
                print(f"   ⚠️  HTTP {exc.code}（临时限流/高负载），{wait_sec}s 后第 {attempt + 1}/{max_retries} 次重试...")
                time.sleep(wait_sec)
                continue
            raise RuntimeError(f"HTTP {exc.code} 调用失败: {body}") from exc
    raise RuntimeError("重试耗尽，调用失败")


def call_mock_llm(messages: list[dict[str, Any]], step: int) -> dict[str, Any]:
    """
    离线 Mock 模型：完整模拟 4 轮真实对话，演示：
    - 第 1 轮：故意尝试读一个不存在的文件（触发 Tool Error）+ 列出笔记
    - 第 2 轮：看到报错和真实文件名后，并行读取两篇真实笔记
    - 第 3 轮：把读到的数字交给 calculate 精确计算
    - 第 4 轮：不再调工具，给出最终带引用与预警结论的回答
    """
    if step == 1:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_mock_1",
                                "type": "function",
                                "function": {
                                    "name": "read_note",
                                    "arguments": json.dumps({"filename": "budget.md"}),
                                },
                            },
                            {
                                "id": "call_mock_2",
                                "type": "function",
                                "function": {"name": "list_notes", "arguments": "{}"},
                            },
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 220, "completion_tokens": 35, "total_tokens": 255},
        }
    if step == 2:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_mock_3",
                                "type": "function",
                                "function": {
                                    "name": "read_note",
                                    "arguments": json.dumps({"filename": "q3_budget.md"}),
                                },
                            },
                            {
                                "id": "call_mock_4",
                                "type": "function",
                                "function": {
                                    "name": "read_note",
                                    "arguments": json.dumps({"filename": "server_costs.md"}),
                                },
                            },
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 340, "completion_tokens": 42, "total_tokens": 382},
        }
    if step == 3:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_mock_5",
                                "type": "function",
                                "function": {
                                    "name": "calculate",
                                    "arguments": json.dumps(
                                        {"expression": "(8200 + 2400 + 1800 + 2600) * 3"}
                                    ),
                                },
                            },
                            {
                                "id": "call_mock_6",
                                "type": "function",
                                "function": {
                                    "name": "calculate",
                                    "arguments": json.dumps(
                                        {"expression": "(8200 + 2400 + 1800 + 2600) * 3 / 48000 * 100"}
                                    ),
                                },
                            },
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 510, "completion_tokens": 58, "total_tokens": 568},
        }
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        "根据本地笔记核算结果如下：\n"
                        "1. **数据来源**：`q3_budget.md`（Q3 总预算 48,000 元，预警线 80%）与 "
                        "`server_costs.md`（月度合计 8200 + 2400 + 1800 + 2600 = 15,000 元/月）。\n"
                        "2. **季度总开销**：15,000 × 3 = **45,000 元**。\n"
                        "3. **预算占比**：45,000 / 48,000 × 100% = **93.75%**。\n"
                        "4. **风险提示**：已超过 80% 预警阈值，建议提前汇报！"
                    ),
                }
            }
        ],
        "usage": {"prompt_tokens": 620, "completion_tokens": 110, "total_tokens": 730},
    }


# =============================================================================
# 5. 核心 Agent Loop（这就是所有 Agent 框架最底层的骨架）
# =============================================================================

def run_agent(
    user_query: str,
    max_steps: int = 6,
    mock: bool = False,
    dump_messages: bool = False,
) -> str:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    total_tokens = 0

    print(f"👤 User: {user_query}")
    print(f"⚙️  Mode: {'MOCK (离线协议演示)' if mock else os.environ.get('LLM_MODEL', 'UNKNOWN')} | max_steps={max_steps}")
    print("-" * 72)

    for step in range(1, max_steps + 1):
        response = (
            call_mock_llm(messages, step)
            if mock
            else call_openai_compatible(messages, TOOLS)
        )

        usage = response.get("usage") or {}
        step_tokens = usage.get("total_tokens", 0)
        total_tokens += step_tokens

        assistant_msg = response["choices"][0]["message"]
        tool_calls = assistant_msg.get("tool_calls") or []

        # 关键协议细节：必须把 assistant 本轮消息原封不动追加入 messages
        messages.append(assistant_msg)

        if not tool_calls:
            final_text = assistant_msg.get("content") or ""
            print(f"\n✅ [Step {step}] Agent 给出最终回答（本轮 {step_tokens} tokens，累计 {total_tokens} tokens）:\n")
            print(final_text)
            if dump_messages:
                print("\n" + "=" * 72)
                print("📦 完整 messages 协议栈（可直接对比 LangChain4j 底层报文）:")
                print(json.dumps(messages, ensure_ascii=False, indent=2))
            return final_text

        print(f"🤖 [Step {step}] 模型请求调用 {len(tool_calls)} 个工具（本轮 {step_tokens} tokens）:")
        for tc in tool_calls:
            tc_id = tc["id"]
            fn_name = tc["function"]["name"]
            fn_args = tc["function"].get("arguments", "{}")
            print(f"   → Tool Call [{tc_id}]: {fn_name}({fn_args})")

            result_json = execute_tool(fn_name, fn_args)
            preview = result_json if len(result_json) <= 140 else result_json[:140] + "..."
            print(f"   ← Tool Result [{tc_id}]: {preview}")

            # 关键协议细节：每个 tool_call 都必须有一条对应 tool_call_id 的 role="tool" 消息
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_json,
                }
            )

    print(f"\n🛑 触发安全护栏：已达最大步数 max_steps={max_steps}（累计 {total_tokens} tokens），强制停止循环。")
    if dump_messages:
        print(json.dumps(messages, ensure_ascii=False, indent=2))
    return ""


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Stage 3: 脱框架手写第一个 Agent Loop")
    parser.add_argument(
        "--query",
        default="帮我查一下笔记里 Q3 预算和各环境服务器月度费用，算一下一整个季度（3个月）的服务器总开销占 Q3 预算的百分之多少？有没有触发预警？",
        help="给 Agent 的任务指令",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="最大循环步数上限")
    parser.add_argument("--mock", action="store_true", help="离线 Mock 模式（无需 API Key）")
    parser.add_argument("--dump-messages", action="store_true", help="结束时打印完整 messages JSON")
    args = parser.parse_args()

    try:
        run_agent(
            user_query=args.query,
            max_steps=args.max_steps,
            mock=args.mock,
            dump_messages=args.dump_messages,
        )
        return 0
    except Exception as exc:
        print(f"❌ 运行出错: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
