"""RAG 安全上下文（LangChain 实现）：检索相似安全示例，拼装防御性 prompt。

企业级 RAG 流程：

1. **Embeddings** —— ``LocalEmbeddings`` 适配本地 OpenAI 兼容的 ``/v1/embeddings``；
2. **VectorStore** —— ``langchain_community.vectorstores.FAISS`` 承载向量索引；
3. **Retriever** —— ``store.similarity_search_with_score(...)`` 做相似度检索，返回
   ``Document`` + 距离，按 ``RAG_MAX_DISTANCE`` 过滤并去重。

持久化用 FAISS 官方格式（索引目录下的 ``index.faiss`` / ``index.pkl``）加一份
``meta.json``（源 JSON 指纹 + 文本条目）。首次运行从 JSON 构建；之后直接加载索引目录；
改了源 JSON（指纹不一致）会自动重建；旧版自研 ``.pt`` 缓存会被读取并**迁移**成新格式。
索引文件由本项目自己生成、存放于仓库目录，沿 FAISS 官方约定以
``allow_dangerous_deserialization=True`` 加载（等同于信任本地缓存文件）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

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
    """安全上下文检索系统：LangChain Embeddings + FAISS 向量库 + Retriever。"""

    def __init__(
        self,
        db_path: Optional[str] = None,
        embed_path: Optional[str] = None,
        api_url: Optional[str] = None,
        model_name: Optional[str] = None,
        *,
        embeddings: Optional[Embeddings] = None,
        index_dir: Optional[str] = None,
        max_distance: Optional[float] = None,
    ) -> None:
        """初始化系统。

        ``embeddings`` 可注入自定义 ``Embeddings``（测试用桩）；缺省使用
        ``LocalEmbeddings`` 指向配置里的本地 embedding 服务。
        ``index_dir`` 为 FAISS 索引目录（缺省取 ``Settings.rag_index_dir``）；
        ``max_distance`` 为检索距离上限（0 = 不过滤）。
        """
        settings = get_settings()
        self.db_path = str(db_path or settings.rag_db_path)
        self.embed_path = str(embed_path or settings.rag_embed_path)
        self.index_dir = self._resolve_index_dir(index_dir, db_path, embed_path, settings)
        self.max_distance = float(
            settings.rag_max_distance if max_distance is None else max_distance
        )
        self.api_url = api_url or settings.embed_api_url
        self.model_name = model_name or settings.embed_model_name

        self.embeddings: Embeddings = embeddings or LocalEmbeddings(
            api_url=self.api_url, model_name=self.model_name
        )
        # 知识库条目：[{"prompt": 攻击样例, "response": 安全回答}]
        self.knowledge_base: List[Dict] = []
        # 与 knowledge_base 一一对应的向量矩阵：只在「全量构建 / 读旧版缓存」时存在，
        # 从 FAISS 索引目录加载时为 None（索引里已含向量）
        self._embedding_matrix: Optional[Any] = None
        self._store: Optional[VectorStore] = None

        if self.index_exists():
            logger.info("检测到已有 FAISS 索引目录，直接加载 ...")
            self.load_index()
        elif os.path.exists(self.embed_path):
            logger.info("检测到旧版 .pt 缓存 %s，读取并迁移为 FAISS 索引目录 ...", self.embed_path)
            self.load_knowledge_base()
        else:
            logger.info("检测到首次运行，开始全量构建向量数据库 ...")
            self.build_vector_db_from_json()

    # ── 向量库构建 / 加载 ────────────────────────────────────────────

    def _resolve_index_dir(
        self,
        index_dir: Optional[str],
        db_path: Optional[str],
        embed_path: Optional[str],
        settings: Any,
    ) -> str:
        """确定 FAISS 索引目录：显式参数 > 随知识库 JSON 派生 > 配置默认值。

        调用方显式指定 ``db_path`` / ``embed_path``（测试、多知识库场景）时索引目录跟着
        知识库 JSON 派生，避免误用全局配置目录、把不同知识库的索引写到一起。
        """
        if index_dir is not None:
            return str(index_dir)
        if db_path is not None or embed_path is not None:
            base = Path(self.db_path)
            return str(base.parent / f"{base.stem}_index")
        return str(settings.rag_index_dir)

    def index_exists(self) -> bool:
        """FAISS 索引目录是否已就绪（以索引文件与元数据都存在为准）。"""
        return os.path.exists(os.path.join(self.index_dir, "index.faiss")) and os.path.exists(
            os.path.join(self.index_dir, "meta.json")
        )

    def build_vector_db_from_json(self, max_prompt_chars: int = 8192) -> None:
        """从 JSON 全量构建向量索引并落盘缓存。"""
        if not os.path.exists(self.db_path):
            logger.error("未找到原始数据文件 %s，无法构建知识库。", self.db_path)
            return

        with open(self.db_path, "r", encoding="utf-8") as f:
            raw_db = json.load(f)

        if not raw_db:
            logger.warning("JSON 数据为空，无需构建。")
            return

        logger.info("正在为 %d 条数据构建向量索引 ...", len(raw_db))

        # 截断过长的 prompt，避免单条请求过大
        truncated_prompts = [item["prompt"][:max_prompt_chars] for item in raw_db]
        vectors = self.embeddings.embed_documents(truncated_prompts)

        self.knowledge_base = [
            {"prompt": item["prompt"], "response": item["safe_response"]} for item in raw_db
        ]
        import torch

        self._embedding_matrix = torch.tensor(vectors, dtype=torch.float32)

        self._build_index()
        self._persist()
        logger.info("知识库构建完成，已保存至 %s", self.index_dir)

    def load_knowledge_base(self) -> None:
        """读取旧版自研 ``.pt`` 缓存（含 list[dict] 老格式），随后迁移为 FAISS 索引目录。

        缓存里记录了源 JSON 的指纹（``fingerprint``）；与当前 JSON 不一致（或旧版缓存没有
        指纹）时**重建**，避免改了知识库仍加载陈旧向量；重建失败则回退旧缓存，不阻断启动。
        """
        if not os.path.exists(self.embed_path):
            self.knowledge_base = []
            self._embedding_matrix = None
            self._store = None
            logger.warning("未找到旧版知识库缓存 %s！", self.embed_path)
            return

        import torch

        logger.info("正在加载旧版知识库 %s ...", self.embed_path)
        # 缓存文件由本项目自己写入（含 tensor），故显式关闭 weights_only 限制
        payload = torch.load(self.embed_path, weights_only=False)
        self.knowledge_base, self._embedding_matrix = self._normalize_payload(payload)

        current = self.source_fingerprint()
        cached = payload.get("fingerprint") if isinstance(payload, dict) else None
        if current is not None and cached != current:
            reason = "旧版缓存缺少指纹" if cached is None else "源 JSON 已变更"
            logger.warning("[RAG] %s，改由 %s 重建向量库 ...", reason, self.db_path)
            self._rebuild_from_source()
            return

        self._build_index()
        self._persist()  # 迁移：把旧缓存写成 FAISS 索引目录，下次直接加载
        logger.info("成功加载！当前防御库容量: %d 条安全上下文。", len(self.knowledge_base))

    def load_index(self) -> None:
        """加载 FAISS 索引目录；源 JSON 指纹不一致时按源文件重建。"""
        meta_path = os.path.join(self.index_dir, "meta.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, ValueError) as exc:
            logger.warning("[RAG] 索引元数据 %s 读取失败（%s），改为重建。", meta_path, exc)
            self.knowledge_base = []
            self._store = None
            self._rebuild_from_source()
            return

        self.knowledge_base = list(meta.get("knowledge_base") or [])
        current = self.source_fingerprint()
        cached = meta.get("fingerprint")
        if current is not None and cached != current:
            reason = "索引缺少指纹" if cached is None else "源 JSON 已变更"
            logger.warning("[RAG] %s，改由 %s 重建向量库 ...", reason, self.db_path)
            self._rebuild_from_source()
            return

        self._load_store()
        logger.info("成功加载 FAISS 索引！当前防御库容量: %d 条安全上下文。", len(self.knowledge_base))

    def _load_store(self) -> None:
        """从索引目录装配 FAISS store（索引文件缺失时置空，降级为空库）。"""
        if not os.path.exists(os.path.join(self.index_dir, "index.faiss")):
            self._store = None
            return
        from langchain_community.vectorstores import FAISS

        # 索引由本项目自己生成、放在仓库目录；不接受用户提供的索引文件
        self._store = FAISS.load_local(
            self.index_dir, self.embeddings, allow_dangerous_deserialization=True
        )

    def source_fingerprint(self) -> Optional[str]:
        """源知识库 JSON 的内容指纹（sha256）；源文件不存在时返回 None。"""
        if not os.path.exists(self.db_path):
            return None
        try:
            with open(self.db_path, "rb") as handle:
                return hashlib.sha256(handle.read()).hexdigest()
        except OSError as exc:
            logger.warning("[RAG] 计算 %s 指纹失败: %s", self.db_path, exc)
            return None

    def _rebuild_from_source(self) -> None:
        """缓存过期时按源 JSON 重建；重建失败则回退到已加载的旧缓存。"""
        cached_kb, cached_matrix = self.knowledge_base, self._embedding_matrix
        try:
            self.build_vector_db_from_json()
        except Exception as exc:  # noqa: BLE001 - 重建失败不应阻断启动
            logger.error("[RAG] 重建向量库失败（%s），回退使用旧缓存。", exc)
        if self._store is not None:
            return
        # 重建没产出索引（源文件缺失 / 服务不可达）：回退旧缓存继续服务
        self.knowledge_base, self._embedding_matrix = cached_kb, cached_matrix
        if cached_matrix is not None:
            self._build_index()
        elif cached_kb:
            self._load_store()

    @staticmethod
    def _normalize_payload(payload: object) -> Tuple[List[Dict], Optional[Any]]:
        """把新旧两种缓存格式统一成 ``(知识库条目, 向量矩阵)``。"""
        import torch

        if isinstance(payload, dict) and "vectors" in payload:  # 新格式
            prompts = list(payload.get("prompts", []))
            responses = list(payload.get("responses", []))
            knowledge_base = [
                {"prompt": prompt, "response": response}
                for prompt, response in zip(prompts, responses, strict=False)
            ]
            return knowledge_base, payload["vectors"]

        if isinstance(payload, (list, tuple)):  # 旧格式：list[{"prompt","response","vector"}]
            knowledge_base = [
                {"prompt": item["prompt"], "response": item["response"]} for item in payload
            ]
            vectors = torch.stack([item["vector"] for item in payload]) if payload else None
            return knowledge_base, vectors

        logger.warning("知识库缓存格式无法识别，按空库处理。")
        return [], None

    def _build_index(self) -> None:
        """用内存中的文本 + 向量装配 FAISS 向量库。"""
        if not self.knowledge_base or self._embedding_matrix is None:
            self._store = None
            logger.warning("知识库为空，向量索引未构建。")
            return

        # FAISS 集成目前仍由 langchain-community 提供（独立的 langchain-faiss 要求
        # Python>=3.11，本项目为 3.10），故按 `langchain-community<1` 锁定。
        from langchain_community.vectorstores import FAISS

        text_embeddings = [
            (item["prompt"], vector)
            for item, vector in zip(
                self.knowledge_base, self._embedding_matrix.tolist(), strict=False
            )
        ]
        metadatas = [{"response": item["response"]} for item in self.knowledge_base]
        self._store = FAISS.from_embeddings(text_embeddings, self.embeddings, metadatas=metadatas)

    def _persist(self) -> None:
        """把 FAISS 索引与文本条目写入索引目录（``index.faiss`` / ``index.pkl`` / ``meta.json``）。"""
        if self._store is None:
            return
        os.makedirs(self.index_dir, exist_ok=True)
        self._store.save_local(self.index_dir)
        meta = {
            # 记录源 JSON 指纹：文件变化后下次启动自动重建
            "fingerprint": self.source_fingerprint(),
            "knowledge_base": self.knowledge_base,
        }
        with open(os.path.join(self.index_dir, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False)

    def _build_index_from_texts(self) -> None:
        """按文本现场计算向量并装配索引（追加样本 / 无缓存时的兜底路径）。"""
        if not self.knowledge_base:
            self._store = None
            return
        from langchain_community.vectorstores import FAISS

        metadatas = [{"response": item["response"]} for item in self.knowledge_base]
        self._store = FAISS.from_texts(
            [item["prompt"] for item in self.knowledge_base], self.embeddings, metadatas=metadatas
        )

    def add_new_attack_context(self, attack_type: str, prompt: str, response: str) -> None:
        """动态增量更新：追加新数据及其向量并持久化。"""
        logger.info("正在为新型攻击 [%s] 计算特征向量 ...", attack_type)
        self.knowledge_base.append({"prompt": prompt, "response": response})
        if self._store is None:
            # 之前是空库（或索引不可用）：按文本重建整库索引
            self._build_index_from_texts()
        else:
            # VectorStore 原生追加：只对新增文本做一次 embedding
            from langchain_core.documents import Document

            self._store.add_documents(
                [Document(page_content=prompt, metadata={"response": response})]
            )
        self._embedding_matrix = None  # 索引已含全部向量，矩阵不再维护
        self._persist()
        logger.info("增量更新成功，已持久化至 %s", self.index_dir)

    # ── 检索 / Prompt 构造 ───────────────────────────────────────────

    def retrieve(
        self, user_query: str, top_k: int = 2, max_distance: Optional[float] = None
    ) -> List[Dict]:
        """检索相似安全示例，返回 ``[{"prompt", "response"}]``。

        用 ``similarity_search_with_score`` 拿到距离后按 ``max_distance`` 过滤
        （0 = 不过滤），并按文本去重（同一攻击样例可能在知识库里重复出现）。
        """
        if self._store is None or not self.knowledge_base:
            logger.warning("知识库为空，无法进行检索。")
            return []

        threshold = self.max_distance if max_distance is None else float(max_distance)
        limit = max(1, min(top_k, len(self.knowledge_base)))
        scored = self._store.similarity_search_with_score(user_query, k=limit)

        hits: List[Dict] = []
        seen: set = set()
        for document, distance in scored:
            key = document.page_content.strip()
            if key in seen:
                continue
            if threshold > 0 and distance > threshold:
                logger.info("[RAG] 丢弃超阈值的命中（distance=%.4f > %.4f）", distance, threshold)
                continue
            seen.add(key)
            hits.append(
                {
                    "prompt": document.page_content,
                    "response": document.metadata.get("response", ""),
                }
            )
        return hits

    def construct_safe_prompt(
        self, user_query: str, top_k: int = 1, pre_retrieved_contexts: Optional[List[Dict]] = None
    ) -> str:
        """将检索到的上下文转换为最终字符串。

        传入 ``pre_retrieved_contexts`` 可避免包装函数调用时重复进行 API 请求。
        """
        system_prefix = """Your task is to provide information and assistance in a safe and ethical manner.
You should not provide any information or assistance that could be used for illegal or harmful activities.
Here are some examples of what you should not do:\n\n"""

        # 优先使用已传入的上下文，否则现场检索
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
    db_path: Optional[str] = None,
    embed_path: Optional[str] = None,
    api_url: Optional[str] = None,
    model_name: Optional[str] = None,
) -> RAGPromptResult:
    """便捷的 RAG 安全提示词构建函数。

    自动初始化系统、检索相关安全上下文，并组装成最终可供 LLM 使用的 Prompt。
    """
    rag_kwargs: Dict = {}
    if db_path is not None:
        rag_kwargs["db_path"] = db_path
    if embed_path is not None:
        rag_kwargs["embed_path"] = embed_path
    if api_url is not None:
        rag_kwargs["api_url"] = api_url
    if model_name is not None:
        rag_kwargs["model_name"] = model_name

    rag_system = SCR_RAG_System(**rag_kwargs)

    retrieved_contexts = rag_system.retrieve(user_query, top_k=top_k)
    final_prompt = rag_system.construct_safe_prompt(
        user_query=user_query, pre_retrieved_contexts=retrieved_contexts
    )

    return RAGPromptResult(
        original_query=user_query,
        final_prompt=final_prompt,
        retrieved_contexts=retrieved_contexts,
    )
