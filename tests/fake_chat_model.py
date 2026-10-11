"""测试用 LangChain 假 ChatModel：记录消息与调用参数，按预设回复返回，不发网络请求。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class RecordingChatModel(BaseChatModel):
    """把每次调用的 messages / kwargs 记进 ``calls``，并弹出 ``replies`` 作为回复。"""

    replies: List[str] = Field(default_factory=list)
    logprob_values: List[Optional[float]] = Field(default_factory=list)
    raise_exc: Any = None
    calls: List[Dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording-stub"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append({"messages": list(messages), "kwargs": dict(kwargs)})
        if self.raise_exc is not None:
            raise self.raise_exc

        content = self.replies.pop(0) if self.replies else ""
        metadata: Dict[str, Any] = {}
        if self.logprob_values:
            logprob = self.logprob_values.pop(0)
            if logprob is not None:
                # 与 OpenAI 兼容服务返回的 choices[0].logprobs 结构一致
                metadata["logprobs"] = {
                    "content": [{"token": content[:4] or "tok", "logprob": logprob, "top_logprobs": []}]
                }
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content, response_metadata=metadata))])


def prompt_text(messages: Sequence[BaseMessage]) -> str:
    """把消息列表拼成纯文本，便于断言 prompt 内容。"""
    parts = []
    for message in messages:
        content = message.content
        parts.append(content if isinstance(content, str) else str(content))
    return "\n".join(parts)
