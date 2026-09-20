"""API 依赖。"""

from __future__ import annotations

from fastapi import HTTPException, Request

from ..services.state import AppState


def get_state(request: Request) -> AppState:
    state: AppState | None = getattr(request.app.state, "app_state", None)
    if state is None or not state.ready:
        raise HTTPException(status_code=503, detail="服务正在初始化，请稍后重试")
    return state
