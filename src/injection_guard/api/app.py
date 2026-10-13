"""防御 API 的应用工厂。"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..config import get_settings
from ..logging_config import setup_logging
from ..persistence.database import get_database
from .auth_routes import router as auth_router
from .dependencies import get_defense_service
from .routes import router
from .session_routes import router as sessions_router

logger = logging.getLogger(__name__)


class SPAStaticFiles(StaticFiles):
    """托管前端构建产物；未知路径回退到 ``index.html``（支持前端路由刷新）。"""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise


def _validation_error_message(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "请求体校验失败"
    first = errors[0]
    loc = ".".join(str(part) for part in first.get("loc", []) if part != "body")
    msg = first.get("msg", "校验失败")
    return f"请求体校验失败: {loc} {msg}".strip()


def create_app() -> FastAPI:
    """构造 FastAPI 应用。"""
    settings = get_settings()
    setup_logging(settings.log_level)

    # 建会话持久化表（幂等）。失败即启动失败，避免运行期才发现存储不可用。
    database = get_database()
    database.create_all()
    logger.info("会话数据库就绪: %s", database.database_url)

    if settings.jwt_secret_key.startswith("insecure-dev-secret-change-me"):
        logger.warning(
            "JWT_SECRET_KEY 仍为默认值，请在生产环境显式设置（openssl rand -hex 32）"
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # 启动时同步预热输入检测器池（含 BERT 分类器）：服务开始接客前模型已就绪，
        # 避免首个检测请求同步等待加载。放到线程里执行，不阻塞事件循环。
        if settings.warmup_on_startup:
            await asyncio.to_thread(get_defense_service().warmup)
        yield

    app = FastAPI(
        title="InjectionGuard · LLM Guardrails API", version=__version__, lifespan=lifespan
    )

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # 把 Pydantic 的 422 统一转成 400 {"error": ...}，保持与旧实现一致的契约
        return JSONResponse(status_code=400, content={"error": _validation_error_message(exc)})

    app.include_router(router)
    app.include_router(auth_router)
    app.include_router(sessions_router)

    # 可选 CORS：逗号分隔的 CORS_ORIGINS；留空则同源部署（默认）。
    origins = [o.strip() for o in (settings.cors_origins or "").split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("已启用 CORS: %s", origins)

    # 托管前端构建产物（若已构建）。挂载在最后：/api/*、/docs 等注册更早，优先命中。
    dist_dir = settings.frontend_dist_dir
    if dist_dir.is_dir():
        app.mount("/", SPAStaticFiles(directory=str(dist_dir), html=True), name="spa")
        logger.info("已托管前端静态资源: %s", dist_dir)

    return app


# 供 `uvicorn injection_guard.api.app:app` 使用
app = create_app()
