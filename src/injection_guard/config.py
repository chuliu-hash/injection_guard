"""集中式配置。

所有可调参数与路径都在此声明，通过环境变量 / ``.env`` 覆盖。
字段名小写下划线，pydantic-settings 默认大小写不敏感映射到环境变量，
因此 ``app_port`` 对应 ``APP_PORT``、``input_max_concurrency`` 对应
``INPUT_MAX_CONCURRENCY``，与旧部署脚本使用的变量名保持兼容。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/injection_guard/config.py -> 仓库根目录
_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── 应用 ──────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8099
    app_debug: bool = False
    app_backlog: int = 128
    log_level: str = "INFO"

    # ── Web 前端静态资源 / 跨域 ────────────────────────────────────────
    # 前端构建产物目录（存在时由 FastAPI 托管 SPA）；dev 走 Vite proxy 时无需构建。
    frontend_dist_dir: Path = _REPO_ROOT / "frontend" / "dist"
    # 逗号分隔的允许跨域来源；留空则不加 CORS 中间件（同源部署无需）。
    cors_origins: Optional[str] = None

    # ── 路径 ──────────────────────────────────────────────────────────
    # 仓库内随代码分发的数据（如宪法规则）与外部模型权重分开管理。
    resources_dir: Path = _REPO_ROOT / "resources"
    model_root: Path = _REPO_ROOT / "models"
    critique_requests_path: Optional[Path] = None  # 默认 resources_dir/CritiqueRequests.json
    bert_model_dir: Optional[Path] = None          # 默认 model_root/Prompt_Injection/model/BERT
    rag_db_path: Optional[Path] = None             # 默认 model_root/Prompt_Injection/overthink_prompt.json
    rag_embed_path: Optional[Path] = None          # 默认 model_root/Prompt_Injection/safety_context_compiled.pt

    # ── 输入检测 ──────────────────────────────────────────────────────
    fixed_sensitivity: str = "low"
    safe_skip_threshold: float = 0.5
    skip_output_when_high_confidence: bool = True
    input_max_concurrency: Optional[int] = None
    output_max_concurrency: Optional[int] = None
    input_acquire_timeout: float = 30.0
    output_acquire_timeout: float = 30.0
    input_detector_gpu_ids: Optional[str] = None
    input_detector_replicas_per_gpu: int = 1

    # 输入检测所调用的本地 LLM（base URL；端点路径由 url_utils.join_endpoint 拼接）
    input_detector_api_url: str = "http://127.1.1.1:8094/v1"
    input_detector_api_model: str = "Qwen3-input"

    # ── 输出检测 ──────────────────────────────────────────────────────
    output_detector_api_url: str = "http://127.1.1.1:8095/v1"
    output_detector_model: str = "Qwen_output"
    output_detector_timeout: float = 60.0
    output_detector_max_tokens: int = 8

    # ── 普通聊天 LLM（用于生成 model_response）───────────────────────
    # 未在请求中提供 model_response 时，流水线调用此接口生成回复。
    chat_api_url: str = "http://127.1.1.1:8097/v1"
    chat_model: str = "Qwen3-chat"
    chat_timeout: float = 120.0
    chat_max_tokens: int = 1024
    chat_temperature: float = 0.7
    chat_max_concurrency: Optional[int] = None
    chat_acquire_timeout: float = 120.0

    # ── Embedding / RAG ───────────────────────────────────────────────
    embed_api_url: str = "http://127.1.1.1:8096/v1"
    embed_model_name: str = "embedding-model"
    embed_gpu_ids: Optional[str] = None
    # RAG 增强：全局开关（默认关），请求体 use_rag 可覆盖
    enable_rag: bool = False
    rag_top_k: int = 2

    # ── 推理服务（services/llm_server.py）────────────────────────────
    # 注意：模型路径 / 模式 / LoRA 等由 configure_model() 在运行时通过环境变量
    # 传入（多 worker 子进程需继承），因此不在此声明。
    api_server_gpu_ids: Optional[str] = None
    api_server_gpu_namespace: Optional[str] = None
    api_server_split_across_gpus: bool = False
    api_server_max_queue_size: int = 32
    api_server_max_input_tokens: int = 2048
    api_server_max_new_tokens: int = 256
    api_server_memory_fraction: Optional[float] = None
    api_server_empty_cache_every: int = 32
    api_server_empty_cache_after_request: bool = False

    # ── 推理服务（services/embed_server.py）──────────────────────────
    embed_gpu_namespace: Optional[str] = None
    embed_request_timeout: float = 60.0

    @field_validator("fixed_sensitivity")
    @classmethod
    def _normalize_sensitivity(cls, value: str) -> str:
        return value.lower()

    def model_post_init(self, __context: object) -> None:
        """补齐派生的路径默认值。"""
        prompt_dir = self.model_root / "Prompt_Injection"
        if self.critique_requests_path is None:
            self.critique_requests_path = self.resources_dir / "CritiqueRequests.json"
        if self.bert_model_dir is None:
            self.bert_model_dir = prompt_dir / "model" / "BERT"
        if self.rag_db_path is None:
            self.rag_db_path = prompt_dir / "overthink_prompt.json"
        if self.rag_embed_path is None:
            self.rag_embed_path = prompt_dir / "safety_context_compiled.pt"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程级单例配置（便于测试通过 ``get_settings.cache_clear()`` 覆盖）。"""
    return Settings()
