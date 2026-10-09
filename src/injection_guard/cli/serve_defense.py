"""启动防御 API（FastAPI + uvicorn）。"""

from __future__ import annotations

import uvicorn

from ..api.app import create_app
from ..config import get_settings
from ..logging_config import setup_logging


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    if settings.app_debug:
        # reload 模式必须使用 import 字符串 + factory
        uvicorn.run(
            "injection_guard.api.app:create_app",
            factory=True,
            host=settings.app_host,
            port=settings.app_port,
            reload=True,
            backlog=settings.app_backlog,
        )
    else:
        uvicorn.run(
            create_app(),
            host=settings.app_host,
            port=settings.app_port,
            backlog=settings.app_backlog,
        )


if __name__ == "__main__":
    main()
