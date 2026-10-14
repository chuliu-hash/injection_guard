"""``DefenseService._run_rag``：命中 / 未命中的状态与 prompt 语义（不加载模型）。"""

from __future__ import annotations

from types import SimpleNamespace

from injection_guard.api.dependencies import DefenseService


class _StubRagSystem:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def retrieve(self, query, top_k=2, max_distance=0.0):
        self.calls.append({"query": query, "top_k": top_k, "max_distance": max_distance})
        return list(self.hits)

    def construct_safe_prompt(self, user_query, pre_retrieved_contexts=None):
        return f"PROMPT:{user_query}"


def _service(rag_system, **settings) -> DefenseService:
    service = DefenseService.__new__(DefenseService)  # 跳过 torch/模型初始化
    service.settings = SimpleNamespace(rag_top_k=2, rag_max_distance=0.5, **settings)
    service._rag_system = rag_system
    return service


def test_run_rag_no_hits_skips_prompt_enhancement():
    """没命中：不拼接上下文，标记 no_match 让生成退化为原始输入。"""
    service = _service(_StubRagSystem([]))

    assert service._run_rag("你好") == {
        "enabled": True,
        "final_prompt": None,
        "reason": "no_match",
        "retrieved_contexts": [],
    }


def test_run_rag_with_hits_builds_prompt_and_passes_threshold():
    rag = _StubRagSystem([{"prompt": "p", "response": "r"}])
    service = _service(rag)

    out = service._run_rag("你好")

    assert out["final_prompt"] == "PROMPT:你好"
    assert out["reason"] is None
    assert out["retrieved_contexts"] == [{"prompt": "p", "response": "r"}]
    # top_k 与距离阈值都来自配置
    assert rag.calls == [{"query": "你好", "top_k": 2, "max_distance": 0.5}]
