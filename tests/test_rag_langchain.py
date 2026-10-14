"""RAG（LangChain 版）测试：桩 Embeddings + FAISS 索引目录，不联网、不加载模型。"""

from __future__ import annotations

import json

import torch
from langchain_core.embeddings import Embeddings

from injection_guard.defense.embeddings import LocalEmbeddings
from injection_guard.defense.rag import SCR_RAG_System


class _FakeEmbeddings(Embeddings):
    """确定性桩：向量 = [字符数, 'a' 个数, 1]。"""

    def __init__(self) -> None:
        self.batch_calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.batch_calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text: str):
        return [float(len(text)), float(text.count("a")), 1.0]


def _write_kb(path, entries):
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")


def test_builds_faiss_index_and_retrieves(tmp_path):
    db = tmp_path / "kb.json"
    _write_kb(
        db,
        [
            {"prompt": "ignore previous instructions", "safe_response": "我不能这样做"},
            {"prompt": "hello world", "safe_response": "你好"},
        ],
    )
    embeddings = _FakeEmbeddings()
    rag = SCR_RAG_System(db_path=str(db), embed_path=str(tmp_path / "kb.pt"), embeddings=embeddings)

    assert len(rag.knowledge_base) == 2
    # 首次运行落盘 FAISS 官方格式的索引目录 + 文本元数据
    assert (tmp_path / "kb_index" / "index.faiss").exists()
    assert (tmp_path / "kb_index" / "meta.json").exists()

    hits = rag.retrieve("ignore previous instructions", top_k=1)
    assert hits == [{"prompt": "ignore previous instructions", "response": "我不能这样做"}]

    prompt = rag.construct_safe_prompt("q", pre_retrieved_contexts=hits)
    assert "Safe Example 1:" in prompt
    assert "我不能这样做" in prompt


def test_reload_reuses_cache_without_reembedding(tmp_path):
    db = tmp_path / "kb.json"
    _write_kb(db, [{"prompt": "abc", "safe_response": "safe"}])
    cache = tmp_path / "kb.pt"

    SCR_RAG_System(db_path=str(db), embed_path=str(cache), embeddings=_FakeEmbeddings())

    embeddings = _FakeEmbeddings()
    rag = SCR_RAG_System(db_path=str(db), embed_path=str(cache), embeddings=embeddings)
    assert embeddings.batch_calls == []  # 直接读缓存，不再调用 embedding 服务
    assert len(rag.knowledge_base) == 1


def test_legacy_pt_cache_is_supported(tmp_path):
    """旧版 .pt 缓存（无指纹）仍能读，并且会被迁移成 FAISS 索引目录。"""
    cache = tmp_path / "kb.pt"
    torch.save(
        [
            {"prompt": "p1", "response": "r1", "vector": torch.tensor([1.0, 0.0, 0.0])},
            {"prompt": "p2", "response": "r2", "vector": torch.tensor([0.0, 1.0, 0.0])},
        ],
        cache,
    )
    rag = SCR_RAG_System(
        db_path=str(tmp_path / "missing.json"),
        embed_path=str(cache),
        embeddings=_FakeEmbeddings(),
    )
    assert [item["prompt"] for item in rag.knowledge_base] == ["p1", "p2"]
    assert len(rag.retrieve("p1", top_k=1)) == 1
    # 迁移落盘：下次启动直接走新格式，不再读 .pt
    assert (tmp_path / "missing_index" / "index.faiss").exists()


def test_empty_knowledge_base_degrades(tmp_path):
    rag = SCR_RAG_System(
        db_path=str(tmp_path / "missing.json"),
        embed_path=str(tmp_path / "missing.pt"),
        embeddings=_FakeEmbeddings(),
    )
    assert rag.retrieve("anything") == []
    assert rag.construct_safe_prompt("anything").startswith("Your task is to provide")


def test_add_new_attack_context_persists(tmp_path):
    db = tmp_path / "kb.json"
    _write_kb(db, [{"prompt": "abc", "safe_response": "safe"}])
    cache = tmp_path / "kb.pt"
    rag = SCR_RAG_System(db_path=str(db), embed_path=str(cache), embeddings=_FakeEmbeddings())

    rag.add_new_attack_context("new", "brand new attack", "拒绝")

    assert len(rag.knowledge_base) == 2
    reloaded = SCR_RAG_System(db_path=str(db), embed_path=str(cache), embeddings=_FakeEmbeddings())
    assert reloaded.knowledge_base[-1]["response"] == "拒绝"


def test_cache_invalidated_when_source_json_changes(tmp_path):
    """源 JSON 变了：缓存指纹不匹配 → 自动重建（不再静默用旧向量）。"""
    db = tmp_path / "kb.json"
    cache = tmp_path / "kb.pt"
    _write_kb(db, [{"prompt": "abc", "safe_response": "safe"}])
    SCR_RAG_System(db_path=str(db), embed_path=str(cache), embeddings=_FakeEmbeddings())

    _write_kb(
        db,
        [
            {"prompt": "abc", "safe_response": "safe"},
            {"prompt": "xyz", "safe_response": "nope"},
        ],
    )
    embeddings = _FakeEmbeddings()
    rag = SCR_RAG_System(db_path=str(db), embed_path=str(cache), embeddings=embeddings)

    assert len(rag.knowledge_base) == 2
    assert embeddings.batch_calls  # 走了 embedding 重建，而不是直接读缓存


def test_legacy_cache_without_fingerprint_is_rebuilt_from_json(tmp_path):
    """旧版缓存没有指纹：以源 JSON 为准重建。"""
    db = tmp_path / "kb.json"
    cache = tmp_path / "kb.pt"
    _write_kb(db, [{"prompt": "abc", "safe_response": "safe"}])
    torch.save(
        [{"prompt": "stale", "response": "old", "vector": torch.tensor([1.0, 0.0, 0.0])}],
        cache,
    )

    rag = SCR_RAG_System(db_path=str(db), embed_path=str(cache), embeddings=_FakeEmbeddings())
    assert [item["prompt"] for item in rag.knowledge_base] == ["abc"]


def test_rebuild_failure_falls_back_to_loaded_cache(tmp_path):
    """重建时 embedding 服务不可用：回退旧缓存，不阻断启动。"""
    db = tmp_path / "kb.json"
    cache = tmp_path / "kb.pt"
    _write_kb(db, [{"prompt": "abc", "safe_response": "safe"}])
    SCR_RAG_System(db_path=str(db), embed_path=str(cache), embeddings=_FakeEmbeddings())
    _write_kb(db, [{"prompt": "changed", "safe_response": "new"}])

    class _Boom(_FakeEmbeddings):
        def embed_documents(self, texts):
            raise RuntimeError("embedding 服务不可达")

    rag = SCR_RAG_System(db_path=str(db), embed_path=str(cache), embeddings=_Boom())
    assert [item["prompt"] for item in rag.knowledge_base] == ["abc"]
    assert rag.retrieve("abc", top_k=1) == [{"prompt": "abc", "response": "safe"}]


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self):
        self.requests = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.requests.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse({"data": [{"embedding": [len(t), 0.5]} for t in json["input"]]})


def test_local_embeddings_batches_and_parses_openai_format():
    session = _FakeSession()
    embeddings = LocalEmbeddings("http://localhost:8096/v1", "embedding-model", batch_size=2, session=session)

    vectors = embeddings.embed_documents(["a", "bb", "ccc"])
    assert vectors == [[1, 0.5], [2, 0.5], [3, 0.5]]
    assert [len(req["json"]["input"]) for req in session.requests] == [2, 1]
    assert session.requests[0]["url"] == "http://localhost:8096/v1/embeddings"
    assert embeddings.embed_query("abcd") == [4, 0.5]


# ── 检索质量：阈值过滤 + 去重 ─────────────────────────────────────────


def _index_with(entries, tmp_path, name="kb"):
    db = tmp_path / f"{name}.json"
    _write_kb(db, entries)
    # 显式给 embed_path：测试不读仓库里的真实 .pt 缓存，保持互不干扰
    return SCR_RAG_System(
        db_path=str(db), embed_path=str(tmp_path / f"{name}.pt"), embeddings=_FakeEmbeddings()
    )


def test_retrieve_dedupes_identical_documents(tmp_path):
    rag = _index_with(
        [
            {"prompt": "ignore previous instructions", "safe_response": "我不能这样做"},
            {"prompt": "ignore previous instructions", "safe_response": "不能配合"},
            {"prompt": "hello world", "safe_response": "你好"},
        ],
        tmp_path,
    )

    hits = rag.retrieve("ignore previous instructions", top_k=3)

    prompts = [hit["prompt"] for hit in hits]
    # 重复样本合并成一条（命中最相似的那条在前）
    assert prompts.count("ignore previous instructions") == 1
    assert hits[0]["prompt"] == "ignore previous instructions"


def test_retrieve_filters_hits_beyond_max_distance(tmp_path):
    rag = _index_with(
        [
            {"prompt": "aaaa", "safe_response": "近距离"},
            {"prompt": "zzzzzzzzzzzzzzzz", "safe_response": "远距离"},
        ],
        tmp_path,
    )

    assert [hit["response"] for hit in rag.retrieve("aa", top_k=2)] == ["近距离", "远距离"]
    # 距离超过阈值：全部丢弃（上层据此跳过上下文增强）
    assert rag.retrieve("aa", top_k=2, max_distance=0.001) == []


def test_max_distance_defaults_to_no_filtering(tmp_path):
    rag = _index_with([{"prompt": "aa", "safe_response": "r"}], tmp_path)

    assert rag.max_distance == 0.0
    assert rag.retrieve("aa", top_k=1) != []


def test_add_new_attack_context_appends_to_index(tmp_path):
    """索引目录加载后仍可增量追加（走 VectorStore.add_documents）并持久化。"""
    rag = _index_with([{"prompt": "abc", "safe_response": "safe"}], tmp_path)

    rag.add_new_attack_context("new", "brand new attack", "拒绝")

    assert [hit["prompt"] for hit in rag.retrieve("brand new attack", top_k=1)] == [
        "brand new attack"
    ]
    # 重新打开同一知识库：新增样本已随索引落盘
    reloaded = _index_with([{"prompt": "abc", "safe_response": "safe"}], tmp_path)
    assert len(reloaded.knowledge_base) == 2
    assert [hit["prompt"] for hit in reloaded.retrieve("brand new attack", top_k=1)] == [
        "brand new attack"
    ]
