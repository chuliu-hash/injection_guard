"""LangChain 消息内容处理的小工具。"""

from __future__ import annotations

from typing import Any


def message_text(content: Any) -> str:
    """把 LangChain 消息内容统一成纯文本。

    字符串直接返回；内容块列表（``content_blocks`` / 多模态）只拼接文本块，
    其它类型返回空串。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return ""
