"""集中式配置。

按用途分三份来源：

- **应用配置** ``Settings``（``.env`` + ``configs/services.env``）—— 防御 API、检测、聊天、RAG 等；
  ``.env`` 只放「访问项目 / 资源路径 / 连接模型」必需项，其余按服务分块的调参放 ``configs/services.env``。
- **推理服务生成参数** ``InferenceSettings``（``configs/services.env``，按服务前缀）—— 仅
  ``services/llm_server.py``（``serve_llm``）加载；与 ``Settings`` 读同一文件，但字段带服务前缀
  （``INPUT_`` / ``OUTPUT_`` / ``CHAT_``），见 :meth:`InferenceSettings.for_service`。
- **部署参数** ``DeploySettings``（``configs/deploy.env``）—— 仅 ``cli/deploy.py``（部署命令）加载。

优先级（高 → 低）：``os.environ`` > 对应的 ``configs/*.env`` > ``.env`` > 代码默认值。
``configs/*.env`` 用**绝对路径**读取（不受 CWD 影响）；``.env`` 仍相对 CWD（与旧行为一致）。

字段名小写下划线，pydantic-settings 默认大小写不敏感映射到环境变量，
因此 ``app_port`` 对应 ``APP_PORT``、``input_max_concurrency`` 对应
``INPUT_MAX_CONCURRENCY``，与旧部署脚本使用的变量名保持兼容。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/injection_guard/config.py -> 仓库根目录
_REPO_ROOT = Path(__file__).resolve().parents[2]

# 输入检测敏感度档位（请求体 sensitivity 的合法取值；阈值表见 defense/prompt_detection.py）。
# 放在本模块是因为 API 层（api/dependencies.py）要校验它，而该层不能引入 torch。
SENSITIVITY_LEVELS = ("low", "medium", "high")

# 运行时服务参数文件（绝对路径）：访问/连接项在 .env，各服务的调参/并发/生成参数在此。
_SERVICES_ENV_FILE = _REPO_ROOT / "configs" / "services.env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # 后者覆盖前者：.env（访问/连接/路径）→ configs/services.env（按服务分块的调参），
        # 二者均被 os.environ 覆盖（优先级：os.environ > services.env > .env > 默认值）。
        env_file=(".env", _SERVICES_ENV_FILE),
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
    # 以下路径均**显式指定**（不再经 MODEL_ROOT / RESOURCES_DIR 间接派生）。
    # 宪法规则文件（仓库内随代码分发）。
    critique_requests_path: Path = _REPO_ROOT / "resources" / "CritiqueRequests.json"
    # BERT 分类模型目录（其下含 4 个子模型），覆盖：BERT_MODEL_DIR
    bert_model_dir: Path = _REPO_ROOT / "models" / "BERT"
    # RAG 知识库 JSON（每条含 prompt / safe_response），覆盖：RAG_DB_PATH
    rag_db_path: Path = _REPO_ROOT / "resources" / "jailbreak_prompts.json"
    # 向量库缓存：与 rag_db_path 同目录、同主名（扩展名 .pt），**无需指定**；
    # 存在则直接加载，不存在则从 rag_db_path 重构。
    rag_embed_path: Optional[Path] = None

    # ── 输入检测 ──────────────────────────────────────────────────────
    # 输入检测敏感度（low / medium / high）：**默认值**——请求体 sensitivity 可逐次覆盖
    # （前端设置面板提供）；可在 configs/services.env 用 DEFAULT_SENSITIVITY 调整。
    default_sensitivity: str = "low"
    # 「请求时」策略：**默认值**——随请求由前端设置面板覆盖
    # （请求体 safe_skip_threshold / skip_output_when_high_confidence）；不写入 configs/services.env。
    safe_skip_threshold: float = 0.5
    skip_output_when_high_confidence: bool = True
    input_max_concurrency: Optional[int] = None
    output_max_concurrency: Optional[int] = None
    input_acquire_timeout: float = 30.0
    output_acquire_timeout: float = 30.0
    input_detector_gpu_ids: Optional[str] = None
    input_detector_replicas_per_gpu: int = 1

    # 输入检测所调用的本地 LLM（base URL；端点路径由 url_utils.join_endpoint 拼接）
    input_detector_api_url: str = "http://localhost:8094/v1"
    input_detector_api_model: str = "Qwen3-input"

    # ── 输出检测 ──────────────────────────────────────────────────────
    output_detector_api_url: str = "http://localhost:8095/v1"
    output_detector_api_key: Optional[str] = None  # 本地留空即可；云端填 key
    output_detector_model: str = "Qwen_output"
    output_detector_timeout: float = 60.0

    # ── 普通聊天 LLM（用于生成 model_response）───────────────────────
    # 未在请求中提供 model_response 时，流水线调用此接口生成回复。
    # 本地（localhost:8097）或云端（填云厂商 base URL + CHAT_API_KEY）均可。
    chat_api_url: str = "http://localhost:8097/v1"
    chat_api_key: Optional[str] = None
    chat_model: str = "Qwen3-chat"
    chat_timeout: float = 120.0
    # 生成参数：**默认值**——随请求由前端设置面板覆盖（请求体 temperature / max_tokens）；
    # 故不写入 configs/services.env（客户端参数不进配置文件）。
    chat_max_tokens: int = 1024
    chat_temperature: float = 0.7
    chat_max_concurrency: Optional[int] = None
    chat_acquire_timeout: float = 120.0

    # ── Embedding / RAG ───────────────────────────────────────────────
    embed_api_url: str = "http://localhost:8096/v1"
    embed_model_name: str = "embedding-model"
    embed_gpu_ids: Optional[str] = None
    # RAG 增强：「请求时」开关的**默认值**（前端设置/请求体 use_rag 覆盖）；不写入 configs/services.env
    enable_rag: bool = False
    rag_top_k: int = 2

    # ── 推理服务（services/embed_server.py）──────────────────────────
    # 注意：LLM 推理服务（services/llm_server.py）的生成参数在 InferenceSettings，
    # 与 Settings 读同一文件（configs/services.env）但带服务前缀，见文件末尾。
    embed_gpu_namespace: Optional[str] = None
    embed_request_timeout: float = 60.0

    @field_validator("default_sensitivity")
    @classmethod
    def _normalize_sensitivity(cls, value: str) -> str:
        return value.lower()

    def model_post_init(self, __context: object) -> None:
        """补齐派生的路径默认值：向量库缓存与知识库 JSON 同目录、同主名（.pt）。"""
        if self.rag_embed_path is None:
            self.rag_embed_path = self.rag_db_path.with_suffix(".pt")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程级单例配置（便于测试通过 ``get_settings.cache_clear()`` 覆盖）。"""
    return Settings()


# ── 推理服务生成参数（按服务前缀，取自 configs/services.env）────────────
# 仅 services/llm_server.py（serve_llm）使用。与 Settings 读同一文件（configs/services.env，
# 绝对路径，不受 CWD 影响），但字段名带**服务前缀**：输入 INPUT_、输出 OUTPUT_、聊天 CHAT_，
# 例如输入服务的 ``INPUT_API_SERVER_MAX_NEW_TOKENS`` → 字段 ``api_server_max_new_tokens``。
# 注意：模型路径 / 模式 / LoRA / 超时由 configure_model() 在运行时通过环境变量注入
# （多 worker 子进程需继承），不在此声明。


def inference_env_name(service: Optional[str], name: str) -> str:
    """按服务映射推理参数的环境变量名（如 ``service="input", name="API_SERVER_GPU_IDS"``
    → ``INPUT_API_SERVER_GPU_IDS``）；``service`` 为空时不加前缀。"""
    return f"{service.upper()}_{name}" if service else name


# 检测服务（输入 / 输出）与它们的默认判定输出长度（token）——见 InferenceSettings.for_service。
# 判定输出很短（如 ``unsafe\\nS5``），16 足够覆盖换行与 S 编码。
_DETECTION_SERVICES = ("input", "output")
_DETECTION_MAX_NEW_TOKENS = 16


class InferenceSettings(BaseSettings):
    """LLM 推理服务的生成参数（``configs/services.env``，按服务前缀）。

    优先用 :meth:`for_service` 取得按服务前缀读取的实例：

        InferenceSettings.for_service("input")   # 读 INPUT_API_SERVER_*
        InferenceSettings.for_service(None)      # 不加前缀，读 API_SERVER_*

    优先级：``os.environ`` > ``configs/services.env`` > 默认值 —— 因此
    ``serve_llm.py`` 与 ``configure_model()`` 写入的运行时环境变量仍可覆盖本文件。
    """

    model_config = SettingsConfigDict(
        env_file=_SERVICES_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_server_gpu_ids: Optional[str] = None
    api_server_gpu_namespace: Optional[str] = None
    api_server_split_across_gpus: bool = False
    api_server_max_queue_size: int = 32
    # 输入编码截断上限（token）；三个检测/聊天服务一致，见 configs/services.env.example
    api_server_max_input_tokens: int = 4096
    # 单次生成 token 上限。**按服务不同**（见 for_service）：输入/输出检测 = 16（只输出判定，
    # 内置默认、无需写进 services.env）；聊天 = None（不设上限，长度由请求体 max_tokens 决定）。
    api_server_max_new_tokens: Optional[int] = None
    api_server_memory_fraction: Optional[float] = None
    api_server_empty_cache_every: int = 32
    api_server_empty_cache_after_request: bool = False

    @classmethod
    def for_service(cls, service: Optional[str] = None, env_file: Optional[Path] = None) -> "InferenceSettings":
        """返回按 ``service`` 前缀读取的实例（``env_file`` 仅供测试覆盖）。"""
        prefix = f"{service.upper()}_" if service else ""
        config = SettingsConfigDict(
            env_file=env_file or _SERVICES_ENV_FILE,
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
            env_prefix=prefix,
        )
        # 检测服务的判定输出长度（16）作为内置默认，无需在 services.env 里重复；
        # 聊天服务不设上限（None）。环境变量 / services.env 仍可覆盖。
        namespace: Dict[str, Any] = {
            "model_config": config,
            "__annotations__": {"api_server_max_new_tokens": Optional[int]},
            "api_server_max_new_tokens": (
                _DETECTION_MAX_NEW_TOKENS if service in _DETECTION_SERVICES else None
            ),
        }
        subclass = type(f"{cls.__name__}_{service or 'default'}", (cls,), namespace)
        return subclass()


@lru_cache(maxsize=None)
def get_inference_settings(service: Optional[str] = None) -> InferenceSettings:
    """返回推理服务配置单例（按服务缓存；测试可用 ``get_inference_settings.cache_clear()``）。"""
    return InferenceSettings.for_service(service)


# ── 部署参数（独立于应用配置）─────────────────────────────────────────
# 仅 cli/deploy.py（injection-guard-deploy 命令）使用。从仓库根 configs/deploy.env
# 读取，用**绝对路径**以免受启动时 CWD 影响。裸变量名（HOST / MODE / WORKERS /
# GPU_IDS）与 shell 环境可能同名，os.environ 会覆盖本文件，可作临时覆盖手段。
_DEPLOY_ENV_FILE = _REPO_ROOT / "configs" / "deploy.env"


class DeploySettings(BaseSettings):
    """本地部署参数（``configs/deploy.env``）。

    各模型路径**显式指定**（不做派生），缺省即用下方默认值。
    """

    model_config = SettingsConfigDict(
        env_file=_DEPLOY_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 模型路径（显式指定，按部署位置修改；不做根目录派生）
    input_model_path: Path = Path("./models/LLM/Qwen_prompt_injection")
    output_model_path: Path = Path("./models/LLM/qwen3-0.6b")
    embed_model_path: Path = Path("./models/Embedding/Qwen3-Embedding-0.6B")
    chat_model_path: Path = Path("./models/LLM/qwen3-0.6b")

    # 绑定地址与访问地址（绑定地址与访问地址分离，勿用 0.0.0.0 作 url_host）
    host: str = "localhost"
    url_host: str = "localhost"

    # 端口（端口单一来源，deploy 命令据此派生 APP_PORT 与各 *_API_URL）
    defense_port: int = 8099
    input_port: int = 8094
    output_port: int = 8095
    embed_port: int = 8096
    chat_port: int = 8097

    # 推理服务参数
    workers: int = 1
    mode: str = "full"
    request_timeout: int = 300
    gpu_ids: Optional[str] = None

    # 是否本地部署 Embedding（否则用 .env 里配置的 EMBED_API_URL，或 RAG 关闭时无需）
    deploy_local_embed: bool = True
    # 是否本地部署聊天 LLM（否则用 .env 里的云端 CHAT_API_URL / CHAT_API_KEY）
    deploy_local_chat: bool = False


@lru_cache(maxsize=1)
def get_deploy_settings() -> DeploySettings:
    """返回部署配置单例（测试可用 ``get_deploy_settings.cache_clear()`` 覆盖）。"""
    return DeploySettings()
