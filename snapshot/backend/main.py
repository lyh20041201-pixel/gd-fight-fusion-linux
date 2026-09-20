"""FastAPI 入口。

生产运行：构建前端后由本服务同时提供 API 与静态页面，只暴露 http://127.0.0.1:8000
开发运行：Vite 位于 127.0.0.1:5173，通过 CORS + 代理访问本服务。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import api_router, ws_router
from .config.settings import get_settings
from .services.logging_setup import setup_logging
from .services.state import AppState

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.logs_dir, settings.log_level)
    logger.info("正在启动教室环境感知与应急控制平台 ...")
    state = AppState(settings)
    app.state.app_state = state
    try:
        await state.start()
    except Exception:  # noqa: BLE001 - 启动失败也要给出可诊断信息
        logger.exception("启动失败")
        raise
    logger.info(
        "启动完成，访问 http://%s:%d （模拟模式=%s）",
        settings.app_host,
        settings.app_port,
        settings.simulation_mode,
    )
    try:
        yield
    finally:
        logger.info("正在关闭，释放摄像头与串口 ...")
        await state.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="教室环境感知与应急控制平台",
        description="本地运行的教室总控平台：环境感知、多路视觉、风险事件与应急报警。",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    app.include_router(ws_router)

    dist = settings.frontend_dist
    if dist.exists():
        assets = dist / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(dist / "index.html")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa(full_path: str):
            """前端路由回退到 index.html；未知 API 路径返回 404 JSON。"""
            if full_path.startswith(("api/", "ws/")):
                return JSONResponse({"detail": "Not Found"}, status_code=404)
            candidate = dist / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")
    else:

        @app.get("/", include_in_schema=False)
        async def dev_index() -> JSONResponse:
            return JSONResponse(
                {
                    "message": "前端尚未构建。开发模式请访问 http://127.0.0.1:5173，"
                    "或在 frontend/ 目录执行 npm run build 后由本服务托管。",
                    "docs": "/docs",
                    "health": "/api/health",
                }
            )

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
