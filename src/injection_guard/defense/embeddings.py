"""LangChain ``Embeddings`` 适配器：对接本地 OpenAI 兼容的 embedding 服务。

本地 embedding 服务（``scripts/serve_embed.sh`` / ``/v1/embeddings``）返回标准
OpenAI 响应格式，这里把它包成 LangChain 的 ``Embeddings`` 接口，供向量库 /
检索器直接使用（批量请求 + 连接池复用）。
"""

from __future__ import annotations

import logging
from typing import List, Optional

import requests
from langchain_core.embeddings import Embeddings

from ..url_utils import join_endpoint

logger = logging.getLogger(__name__)


class LocalEmbeddings(Embeddings):
    """把 ``POST {api_url}/embeddings`` 包装成 LangChain Embeddings。"""

    def __init__(
        self,
        api_url: str,
        model_name: str,
        *,
        batch_size: int = 16,
        timeout: float = 60.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_url = api_url
        self.model_name = model_name
        self.batch_size = batch_size
        self.timeout = timeout
        if session is not None:
            self._http_session = session
        else:
            self._http_session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
            self._http_session.mount("http://", adapter)
            self._http_session.mount("https://", adapter)

    # ── LangChain Embeddings 接口 ────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化（按 ``batch_size`` 切分，避免单次请求过大）。"""
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """单条查询向量化。"""
        return self._embed_batch([text])[0]

    # ── 内部实现 ─────────────────────────────────────────────────────

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        payload = {"input": texts, "model": self.model_name}
        try:
            response = self._http_session.post(
                join_endpoint(self.api_url, "/embeddings"),
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            # OpenAI 格式：数据在 "data" 数组，每个元素的 "embedding" 字段是向量
            return [item["embedding"] for item in response.json()["data"]]
        except requests.exceptions.RequestException as exc:
            logger.error("调用 Embedding API 失败: %s", exc)
            raise
