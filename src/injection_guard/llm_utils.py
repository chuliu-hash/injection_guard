"""LangChain 消息内容处理的小工具。"""

from __future__ import annotations

from typing import Any


def estimate_tokens(text: str) -> int:
    """粗略估算 ``text`` 的 token 数（供流式增量输出检测的触发阈值使用）。

    不绑定任何具体模型的分词器：中日韩/全角字符按 1 字 ≈ 1 token，其余按 4 字符 ≈ 1 token
    估算，非空文本至少返回 1。真实词表（Qwen / BERT / GPT）各不相同，这里只用于判断
    「是否够一个间隔」，不作为计费、截断或长度限制的依据。
    """
    if not text:
        return 0
    wide = sum(
        1
        for ch in text
        if "\u2e80" <= ch <= "\u9fff" or "\uf900" <= ch <= "\ufaff" or "\uff00" <= ch <= "\uffef"
    )
    narrow = len(text) - wide
    return max(1, wide + (narrow + 3) // 4)


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
