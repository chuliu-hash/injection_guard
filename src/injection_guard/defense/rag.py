"""RAG 安全上下文：使用 Milvus 检索相似安全示例，拼装防御性 prompt。

流程：

1. ``LocalEmbeddings`` 调用本地 OpenAI 兼容的 ``/v1/embeddings``；
2. ``MilvusClient`` 负责向量存储与相似度检索；
3. 检索结果按 ``RAG_MAX_DISTANCE`` 过滤并去重后注入安全 prompt。

Milvus 中保存 ``vector`` / ``prompt`` / ``response`` / ``position`` 字段。启动时读取
知识库目录下的全部 JSON，并按文件内容指纹判断是否需要重建集合；未变化时直接复用
集合，不重复调用 embedding 服务。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings

from ..config import get_settings
from .embeddings import LocalEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class RAGPromptResult:
    """结构化的 RAG 返回结果。"""

    original_query: str  # 用户原始查询
    final_prompt: str  # 拼接防注入上下文后的最终 Prompt
    retrieved_contexts: List[Dict]  # 检索命中的相关安全示例明细


class SCR_RAG_System:
    """安全上下文检索系统：LangChain Embeddings + Milvus。"""

    def __init__(
        self,
        kb_dir: Optional[str] = None,
        api_url: Optional[str] = None,
        model_name: Optional[str] = None,
        *,
        embeddings: Optional[Embeddings] = None,
        milvus_uri: Optional[str] = None,
        collection_name: Optional[str] = None,
        meta_path: Optional[str] = None,
        max_distance: Optional[float] = None,
        client: Optional[Any] = None,
    ) -> None:
        """初始化系统。

        ``embeddings`` 与 ``client`` 可注入测试桩；缺省分别使用本地 embedding 服务与
        ``MilvusClient(uri=settings.milvus_uri)``。
        """
        settings = get_settings()
        self.kb_dir = Path(kb_dir or settings.rag_dir).expanduser()
        self.meta_path = self._resolve_meta_path(meta_path, kb_dir, settings)
        self.milvus_uri = milvus_uri or settings.milvus_uri
        self.collection_name = collection_name or settings.milvus_collection
        self.max_distance = float(
            settings.rag_max_distance if max_distance is None else max_distance
        )
        self.api_url = api_url or settings.embed_api_url
        self.model_name = model_name or settings.embed_model_name

        self.embeddings: Embeddings = embeddings or LocalEmbeddings(
            api_url=self.api_url, model_name=self.model_name
        )
        self._client: Any = client or self._build_client()
        self.knowledge_base: List[Dict[str, str]] = []
        self._collection_ready = False

        self._initialize()

    def _build_client(self) -> Any:
        from pymilvus import MilvusClient

        return MilvusClient(uri=self.milvus_uri)

    def _resolve_meta_path(self, meta_path: Optional[str], kb_dir: Optional[str], settings: Any) -> str:
        """显式参数 > 随显式知识库目录派生 > 全局配置。"""
        if meta_path is not None:
            return str(meta_path)
        if kb_dir is not None:
            return str(self.kb_dir.parent / f"{self.kb_dir.name}_milvus.json")
        return str(settings.rag_meta_path)

    # ── 集合初始化 / 重建 ────────────────────────────────────────────

    def _initialize(self) -> None:
        """复用指纹一致的集合；源 JSON 变化或集合缺失时重建。"""
        fingerprint = self.source_fingerprint()
        if self._has_collection():
            self._load_records_from_milvus()
            self._collection_ready = True
            if self._meta_matches(fingerprint):
                logger.info(
                    "复用 Milvus 集合 %s（%d 条安全上下文）",
                    self.collection_name,
                    len(self.knowledge_base),
                )
                return
            logger.info("Milvus 集合指纹不一致，按源知识库重建 ...")
        else:
            logger.info("Milvus 集合 %s 不存在，开始构建 ...", self.collection_name)

        if fingerprint is None:
            logger.warning("知识库目录 %s 不存在，Milvus 集合保持为空。", self.kb_dir)
            return

        try:
            self.build_vector_db_from_json()
        except Exception as exc:  # noqa: BLE001 - 重建失败不应阻断服务启动
            logger.error("[RAG] 重建 Milvus 集合失败（%s），尝试复用旧集合。", exc)
            if self._collection_ready:
                return
            self.knowledge_base = []

    def _has_collection(self) -> bool:
        return bool(self._client.has_collection(collection_name=self.collection_name))

    def _meta_matches(self, fingerprint: Optional[str]) -> bool:
        meta = self._read_meta()
        return (
            meta.get("collection_name") == self.collection_name
            and "fingerprint" in meta
            and meta["fingerprint"] == fingerprint
        )

    def _read_meta(self) -> Dict[str, Any]:
        try:
            with open(self.meta_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_meta(self) -> None:
        Path(self.meta_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "fingerprint": self.source_fingerprint(),
                    "collection_name": self.collection_name,
                },
                handle,
                ensure_ascii=False,
            )

    def _json_files(self) -> List[Path]:
        """按相对路径排序返回知识库目录内的全部 JSON 文件。"""
        if not self.kb_dir.is_dir():
            return []
        return sorted(
            (path for path in self.kb_dir.rglob("*.json") if path.is_file()),
            key=lambda path: path.relative_to(self.kb_dir).as_posix(),
        )

    def source_fingerprint(self) -> Optional[str]:
        """知识库目录内容指纹（sha256）；目录不存在时返回 None。"""
        if not self.kb_dir.is_dir():
            return None
        digest = hashlib.sha256()
        try:
            for path in self._json_files():
                relative = path.relative_to(self.kb_dir).as_posix()
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
                digest.update(path.read_bytes())
                digest.update(b"\0")
        except OSError as exc:
            logger.warning("[RAG] 计算知识库目录 %s 指纹失败: %s", self.kb_dir, exc)
            return None
        return digest.hexdigest()

    def build_vector_db_from_json(self, max_prompt_chars: int = 8192) -> None:
        """合并知识库目录中的全部 JSON，并全量构建 Milvus 集合。"""
        if not self.kb_dir.is_dir():
            logger.warning("知识库目录 %s 不存在，跳过构建。", self.kb_dir)
            return

        files = self._json_files()
        entries: List[Dict[str, str]] = []
        prompts: List[str] = []
        for path in files:
            relative = path.relative_to(self.kb_dir).as_posix()
            with open(path, "r", encoding="utf-8") as handle:
                raw_db = json.load(handle)
            if not isinstance(raw_db, list):
                raise ValueError(f"知识库文件 {relative} 的根节点必须是 JSON 数组")
            for index, item in enumerate(raw_db):
                if not isinstance(item, dict) or "prompt" not in item or "safe_response" not in item:
                    raise ValueError(
                        f"知识库文件 {relative} 第 {index + 1} 条缺少 prompt / safe_response"
                    )
                prompt = str(item["prompt"])
                prompts.append(prompt[:max_prompt_chars])
                entries.append({"prompt": prompt, "response": str(item["safe_response"])})

        if not entries:
            logger.warning("知识库目录 %s 中没有可用条目，清空 Milvus 集合。", self.kb_dir)
            self._clear_collection()
            return

        logger.info("正在为 %d 条数据计算向量并写入 Milvus ...", len(entries))
        vectors = self.embeddings.embed_documents(prompts)
        if len(vectors) != len(entries):
            raise ValueError("embedding 返回数量与知识库条目数量不一致")

        self._replace_collection(entries, vectors)
        self.knowledge_base = entries
        self._collection_ready = True
        self._write_meta()
        logger.info("Milvus 知识库构建完成: collection=%s, count=%d", self.collection_name, len(entries))

    def _clear_collection(self) -> None:
        """知识库为空时删除旧集合，避免继续召回已移除的条目。"""
        if self._has_collection():
            self._client.drop_collection(collection_name=self.collection_name)
        self.knowledge_base = []
        self._collection_ready = False
        self._write_meta()

    def _replace_collection(self, entries: List[Dict[str, str]], vectors: List[List[float]]) -> None:
        if not vectors:
            return
        if self._has_collection():
            self._client.drop_collection(collection_name=self.collection_name)
        self._create_collection(dimension=len(vectors[0]))
        self._insert_records(entries, vectors, start_position=0)

    def _create_collection(self, dimension: int) -> None:
        self._client.create_collection(
            collection_name=self.collection_name,
            dimension=dimension,
            primary_field_name="id",
            vector_field_name="vector",
            id_type="int",
            auto_id=True,
            metric_type="L2",
        )
        self._client.load_collection(collection_name=self.collection_name)

    def _insert_records(
        self,
        entries: List[Dict[str, str]],
        vectors: List[List[float]],
        *,
        start_position: int,
    ) -> None:
        data = [
            {
                "vector": vector,
                "prompt": entry["prompt"],
                "response": entry["response"],
                "position": start_position + index,
            }
            for index, (entry, vector) in enumerate(zip(entries, vectors, strict=True))
        ]
        self._client.insert(collection_name=self.collection_name, data=data)
        self._client.flush(collection_name=self.collection_name)
        self._client.load_collection(collection_name=self.collection_name)

    def _load_records_from_milvus(self) -> None:
        """读取集合内全部文本字段；向量只留在 Milvus，不进入进程内存。"""
        self._client.load_collection(collection_name=self.collection_name)
        records: List[Dict[str, Any]] = []
        offset = 0
        page_size = 1000
        while True:
            page = self._client.query(
                collection_name=self.collection_name,
                filter="id >= 0",
                output_fields=["prompt", "response", "position"],
                limit=page_size,
                offset=offset,
                consistency_level="Strong",
            )
            if not page:
                break
            records.extend(page)
            if len(page) < page_size:
                break
            offset += len(page)

        records.sort(key=lambda item: int(item.get("position", 0)))
        self.knowledge_base = [
            {"prompt": str(item.get("prompt", "")), "response": str(item.get("response", ""))}
            for item in records
        ]

    def add_new_attack_context(self, attack_type: str, prompt: str, response: str) -> None:
        """动态追加一条安全上下文并立即写入 Milvus。"""
        logger.info("正在为新型攻击 [%s] 计算特征向量 ...", attack_type)
        vector = self.embeddings.embed_documents([prompt])[0]
        entry = {"prompt": prompt, "response": response}

        if not self._collection_ready:
            self._create_collection(dimension=len(vector))
            self._collection_ready = True
            start_position = 0
        else:
            start_position = len(self.knowledge_base)

        self._insert_records([entry], [vector], start_position=start_position)
        self.knowledge_base.append(entry)
        self._write_meta()
        logger.info("增量更新成功: collection=%s, count=%d", self.collection_name, len(self.knowledge_base))

    # ── 检索 / Prompt 构造 ───────────────────────────────────────────

    def retrieve(
        self, user_query: str, top_k: int = 2, max_distance: Optional[float] = None
    ) -> List[Dict]:
        """检索相似安全示例，返回 ``[{"prompt", "response"}]``。"""
        if not self._collection_ready or not self.knowledge_base:
            logger.warning("Milvus 知识库为空，无法进行检索。")
            return []

        threshold = self.max_distance if max_distance is None else float(max_distance)
        limit = max(1, min(top_k, len(self.knowledge_base)))
        query_vector = self.embeddings.embed_query(user_query)
        results = self._client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["prompt", "response"],
            search_params={"metric_type": "L2"},
            consistency_level="Strong",
        )

        hits: List[Dict] = []
        seen: set[str] = set()
        for hit in results[0] if results else []:
            entity = hit.get("entity") or {}
            prompt = str(entity.get("prompt", ""))
            key = prompt.strip()
            if not key or key in seen:
                continue
            distance = float(hit.get("distance", 0.0))
            if threshold > 0 and distance > threshold:
                logger.info("[RAG] 丢弃超阈值的命中（distance=%.4f > %.4f）", distance, threshold)
                continue
            seen.add(key)
            hits.append({"prompt": prompt, "response": str(entity.get("response", ""))})
        return hits

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def construct_safe_prompt(
        self, user_query: str, top_k: int = 1, pre_retrieved_contexts: Optional[List[Dict]] = None
    ) -> str:
        """将检索到的上下文转换为最终字符串。

        传入 ``pre_retrieved_contexts`` 可避免包装函数调用时重复进行向量检索。
        """
        system_prefix = """Your task is to provide information and assistance in a safe and ethical manner.
You should not provide any information or assistance that could be used for illegal or harmful activities.
Here are some examples of what you should not do:\n\n"""

        safe_contexts = (
            pre_retrieved_contexts
            if pre_retrieved_contexts is not None
            else self.retrieve(user_query, top_k)
        )

        context_str = ""
        for i, ctx in enumerate(safe_contexts):
            context_str += (
                f"Safe Example {i + 1}:\nUser: {ctx['prompt']}\nAssistant: {ctx['response']}\n\n"
            )

        final_query = "Now, please answer the following query in a safe and ethical manner:\n"
        return system_prefix + context_str + final_query


def get_safe_rag_prompt(
    user_query: str,
    top_k: int = 2,
    kb_dir: Optional[str] = None,
    api_url: Optional[str] = None,
    model_name: Optional[str] = None,
    milvus_uri: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> RAGPromptResult:
    """便捷的 RAG 安全提示词构建函数。"""
    rag_kwargs: Dict[str, Any] = {}
    if kb_dir is not None:
        rag_kwargs["kb_dir"] = kb_dir
    if api_url is not None:
        rag_kwargs["api_url"] = api_url
    if model_name is not None:
        rag_kwargs["model_name"] = model_name
    if milvus_uri is not None:
        rag_kwargs["milvus_uri"] = milvus_uri
    if collection_name is not None:
        rag_kwargs["collection_name"] = collection_name

    rag_system = SCR_RAG_System(**rag_kwargs)
    try:
        retrieved_contexts = rag_system.retrieve(user_query, top_k=top_k)
        final_prompt = rag_system.construct_safe_prompt(
            user_query=user_query, pre_retrieved_contexts=retrieved_contexts
        )
    finally:
        rag_system.close()

    return RAGPromptResult(
        original_query=user_query,
        final_prompt=final_prompt,
        retrieved_contexts=retrieved_contexts,
    )
