"""普通聊天 LLM 客户端（OpenAI 兼容）。

防御流水线在请求未提供 ``model_response`` 时，用它调用「普通聊天接口」生成回复。
与 ``defense/safety_detector.py`` 一致，走 ``openai`` SDK + ``base_url``。
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from openai import OpenAI

from ..config import get_settings

logger = logging.getLogger(__name__)


class ChatClient:
    """调用 OpenAI 兼容的聊天补全接口，返回纯文本回复。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self.base_url = base_url or settings.chat_api_url
        self.model_name = model_name or settings.chat_model
        # 本地服务无鉴权：CHAT_API_KEY 留空即可；openai SDK 不接受空 key，
        # 故留空时用占位串（本地服务忽略该 Authorization 头），填了则用真 key。
        self.api_key = (api_key if api_key is not None else settings.chat_api_key) or "EMPTY"
        self.timeout = timeout if timeout is not None else settings.chat_timeout
        self.max_tokens = max_tokens if max_tokens is not None else settings.chat_max_tokens
        self.temperature = temperature if temperature is not None else settings.chat_temperature

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def _request_params(
        self,
        max_tokens: Optional[int],
        temperature: Optional[float],
    ) -> dict:
        """按请求覆盖实例默认值（默认值来自 Settings）。"""
        return {
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """发送单轮用户消息，返回模型回复文本。

        ``max_tokens`` / ``temperature`` 为本请求的生成参数；留空时用实例默认值
        （构造时取自 Settings，可由前端设置面板经请求体覆盖）。
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **self._request_params(max_tokens, temperature),
        )
        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice is not None and choice.message else None
        return content or ""

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """流式发送单轮用户消息，逐个 yield 增量文本片段（delta）。

        ``max_tokens`` / ``temperature`` 语义同 :meth:`complete`。
        """
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **self._request_params(max_tokens, temperature),
            stream=True,
        )
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                yield content
