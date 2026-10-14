"""进程内 LLM 指标：回调记录耗时 / token / 失败，并可被 /health 读到。"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from injection_guard.services import llm_metrics
from injection_guard.services.chat_client import ChatClient
from injection_guard.services.llm_metrics import MetricsCallbackHandler


def _result(prompt_tokens: int = 0, completion_tokens: int = 0) -> LLMResult:
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    return LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="hi"))]],
        llm_output={"token_usage": usage},
    )


def setup_function() -> None:
    llm_metrics.reset()


def test_end_callback_records_latency_and_tokens() -> None:
    handler = MetricsCallbackHandler("chat")
    handler.on_llm_start({}, ["hi"], run_id="run-1")
    handler.on_llm_end(_result(3, 5), run_id="run-1")

    stats = llm_metrics.snapshot()["chat"]
    assert stats["calls"] == 1
    assert stats["failures"] == 0
    assert stats["prompt_tokens"] == 3
    assert stats["completion_tokens"] == 5
    assert stats["avg_latency_ms"] >= 0.0


def test_error_callback_records_failure() -> None:
    handler = MetricsCallbackHandler("output_detector")
    handler.on_chat_model_start({}, [["hi"]], run_id="run-2")
    handler.on_llm_error(RuntimeError("down"), run_id="run-2")

    stats = llm_metrics.snapshot()["output_detector"]
    assert stats["calls"] == 1
    assert stats["failures"] == 1
    assert stats["prompt_tokens"] == 0


def test_tokens_fall_back_to_message_usage_metadata() -> None:
    handler = MetricsCallbackHandler("input_detector")
    handler.on_llm_start({}, ["hi"], run_id="run-3")
    message = AIMessage(
        content="ok", usage_metadata={"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}
    )
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id="run-3")

    stats = llm_metrics.snapshot()["input_detector"]
    assert (stats["prompt_tokens"], stats["completion_tokens"]) == (7, 2)


def test_chat_client_wires_the_metrics_callback() -> None:
    client = ChatClient()
    assert any(isinstance(cb, MetricsCallbackHandler) for cb in client.model.callbacks)
