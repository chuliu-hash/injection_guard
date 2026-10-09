"""OpenAI 兼容接口的 URL 拼接工具。

所有下游服务（输入/输出检测、聊天、Embedding）的配置项统一为 **base URL**
（形如 ``http://127.1.1.1:8094/v1``），端点路径由调用处拼接，避免配置风格不一。
"""

from __future__ import annotations


def join_endpoint(base_url: str, path: str) -> str:
    """把 base URL 与 OpenAI 端点路径拼成完整 URL。

    容忍旧的「全路径」配置：若 ``base_url`` 已经以 ``path`` 结尾，则原样返回，
    不会重复拼接（例如 base 已是 ``.../v1/chat/completions``）。

    >>> join_endpoint("http://127.1.1.1:8094/v1", "/chat/completions")
    'http://127.1.1.1:8094/v1/chat/completions'
    >>> join_endpoint("http://127.1.1.1:8094/v1/chat/completions", "/chat/completions")
    'http://127.1.1.1:8094/v1/chat/completions'
    """
    base = (base_url or "").rstrip("/")
    endpoint = path if path.startswith("/") else f"/{path}"
    if base.endswith(endpoint):
        return base
    return f"{base}{endpoint}"
