"""普通聊天 LLM 客户端（LangChain ChatModel，OpenAI 兼容）。

防御流水线在请求未提供 ``model_response`` 时，用它调用「普通聊天接口」生成回复。
实现基于 LangChain：``ChatPromptTemplate`` 组装消息 + ``ChatOpenAI`` 指向本地
OpenAI 兼容服务；``max_tokens`` / ``temperature`` 按请求经 ``bind`` 覆盖。
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..config import get_settings
from ..llm_utils import message_text

logger = logging.getLogger(__name__)

# 单轮用户消息模板（模板变量由 LangChain 负责渲染，用户内容按值插入，不会被二次解析）
CHAT_PROMPT = ChatPromptTemplate.from_messages([("human", "{prompt}")])


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
        # 本地服务无鉴权：CHAT_API_KEY 留空即可；OpenAI 兼容客户端不接受空 key，
        # 故留空时用占位串（本地服务忽略该 Authorization 头），填了则用真 key。
        self.api_key = (api_key if api_key is not None else settings.chat_api_key) or "EMPTY"
        self.timeout = timeout if timeout is not None else settings.chat_timeout
        self.max_tokens = max_tokens if max_tokens is not None else settings.chat_max_tokens
        self.temperature = temperature if temperature is not None else settings.chat_temperature

        self.model = ChatOpenAI(
            model=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def _bind_model(self, max_tokens: Optional[int], temperature: Optional[float]):
        """按请求覆盖生成参数（缺省回落实例默认值）。"""
        params = {
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        return self.model.bind(**params)

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
        messages = CHAT_PROMPT.format_messages(prompt=prompt)
        response = self._bind_model(max_tokens, temperature).invoke(messages)
        return message_text(response.content)

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
        messages = CHAT_PROMPT.format_messages(prompt=prompt)
        for chunk in self._bind_model(max_tokens, temperature).stream(messages):
            text = message_text(chunk.content)
            if text:
                yield text
