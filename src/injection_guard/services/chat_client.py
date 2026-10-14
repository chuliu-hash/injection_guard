"""普通聊天 LLM 客户端（LangChain ChatModel，OpenAI 兼容）。

防御流水线在请求未提供 ``model_response`` 时，用它调用「普通聊天接口」生成回复。
实现基于 LangChain：``ChatPromptTemplate`` + ``MessagesPlaceholder`` 组装
「历史消息 + 本轮提问」，``ChatOpenAI`` 指向本地 OpenAI 兼容服务；``max_tokens`` /
``temperature`` 按请求经 ``bind`` 覆盖。

多轮记忆：``history`` 由调用方（服务端按会话读取，见 ``persistence.repository``）传入，
按轮数截断后再用 ``trim_messages`` 裁到 token 预算内，避免把整段历史塞进上下文。
"""

from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Optional, Sequence

from langchain_core.messages import BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from ..config import get_settings
from ..llm_utils import message_text
from .llm_metrics import MetricsCallbackHandler

logger = logging.getLogger(__name__)

# 聊天消息模板：可选历史（``MessagesPlaceholder``）+ 本轮用户消息。
# 模板变量由 LangChain 负责渲染，用户内容按值插入，不会被二次解析。
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder("history", optional=True), ("human", "{prompt}")]
)

# 历史消息的角色映射：只认 user / assistant（system 由聊天模板自己决定）
_HISTORY_ROLES = {"user": "human", "assistant": "ai"}


def _history_messages(history: Optional[Sequence[Dict[str, str]]]) -> List[Dict[str, str]]:
    """把 ``[{"role": "user"|"assistant", "content": ...}]`` 归一化成模板可渲染的字典。"""
    messages: List[Dict[str, str]] = []
    for item in history or []:
        content = (item.get("content") or "").strip()
        role = _HISTORY_ROLES.get((item.get("role") or "").strip().lower())
        if not content or role is None:
            continue
        messages.append({"role": role, "content": item["content"]})
    return messages


def approximate_tokens(messages: Sequence[BaseMessage]) -> int:
    """粗略的 token 估算（按字符数 / 4）。

    本地 GPTQ 服务的分词器与 OpenAI 不同，用 tiktoken 精确计数反而更偏，故统一用
    可解释的近似值：只用于「历史裁剪」与「增量检测间隔」这类阈值判断。
    """
    return sum(max(1, len(str(message.content)) // 4) for message in messages)


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
        history_max_tokens: Optional[int] = None,
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
        self.history_max_tokens = (
            history_max_tokens if history_max_tokens is not None else settings.chat_history_max_tokens
        )

        self.model = ChatOpenAI(
            model=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            callbacks=[MetricsCallbackHandler("chat")],
        )

    def _bind_model(self, max_tokens: Optional[int], temperature: Optional[float]):
        """按请求覆盖生成参数（缺省回落实例默认值）。"""
        params = {
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        return self.model.bind(**params)

    def _build_messages(self, prompt: str, history: Optional[Sequence[Dict[str, str]]]):
        """渲染「历史 + 本轮提问」，并按 token 预算保留最近的消息。"""
        messages = CHAT_PROMPT.format_messages(prompt=prompt, history=_history_messages(history))
        if self.history_max_tokens > 0 and len(messages) > 1:
            trimmed = trim_messages(
                messages,
                max_tokens=self.history_max_tokens,
                strategy="last",
                token_counter=approximate_tokens,
                allow_partial=False,
            )
            # 单条历史就超预算时 trim_messages 会返回空列表（allow_partial=False 不切消息）：
            # 此时只保留本轮提问，绝不把用户这次的问题也裁掉。
            messages = trimmed or [messages[-1]]
        return messages

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        history: Optional[Sequence[Dict[str, str]]] = None,
    ) -> str:
        """发送单轮用户消息，返回模型回复文本。

        ``max_tokens`` / ``temperature`` 为本请求的生成参数；留空时用实例默认值
        （构造时取自 Settings，可由前端设置面板经请求体覆盖）。
        ``history`` 为之前的对话（按时间顺序），用于多轮上下文。
        """
        messages = self._build_messages(prompt, history)
        response = self._bind_model(max_tokens, temperature).invoke(messages)
        return message_text(response.content)

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        history: Optional[Sequence[Dict[str, str]]] = None,
    ) -> Iterator[str]:
        """流式发送单轮用户消息，逐个 yield 增量文本片段（delta）。

        ``max_tokens`` / ``temperature`` / ``history`` 语义同 :meth:`complete`。
        """
        messages = self._build_messages(prompt, history)
        for chunk in self._bind_model(max_tokens, temperature).stream(messages):
            text = message_text(chunk.content)
            if text:
                yield text
