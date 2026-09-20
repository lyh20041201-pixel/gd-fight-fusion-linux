"""系统状态相关接口。"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from ..schemas.api import ActionResult, SimulateRequest
from ..schemas.common import ModuleStatus, SystemStatus
from ..services.state import AppState
from .deps import get_state

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """轻量健康检查，初始化未完成时也能返回。"""
    state: AppState | None = getattr(request.app.state, "app_state", None)
    if state is None:
        return {"ok": False, "ready": False, "server_time": time.time()}
    return {
        "ok": True,
        "ready": state.ready,
        "simulation_mode": state.runtime.simulation_mode,
        "database_ok": state.db.healthy,
        "uptime_seconds": round(time.time() - state.started_at, 1),
        "server_time": time.time(),
    }


@router.get("/system/status", response_model=SystemStatus)
async def system_status(state: AppState = Depends(get_state)) -> SystemStatus:
    return state.system_status()


@router.get("/modules", response_model=list[ModuleStatus])
async def modules(state: AppState = Depends(get_state)) -> list[ModuleStatus]:
    return state.modules()


@router.get('/actions/status')
async def action_status(state: AppState = Depends(get_state)):
    if state.live_actions is None:
        return dict(enabled=False, state='disabled', reason=None, cameras={})
    result = state.live_actions.status()
    result['cloud_review_enabled'] = state.runtime.qwen_enabled
    return result


@router.get("/logs")
async def logs(
    limit: int = Query(200, ge=1, le=1000),
    level: str | None = Query(None, max_length=16),
    state: AppState = Depends(get_state),
) -> list[dict[str, Any]]:
    return state.repo.list_logs(limit=limit, level=level)


@router.post("/system/simulate", response_model=ActionResult)
async def simulate(
    payload: SimulateRequest, state: AppState = Depends(get_state)
) -> ActionResult:
    """注入模拟场景（仅模拟模式可用），用于演示与验收。"""
    result = await state.simulate(payload.scenario, payload.node_id, payload.camera_id)
    return ActionResult(
        ok=bool(result.get("ok")),
        message=str(result.get("message", "已注入模拟场景")),
        detail=result.get("detail", {}),
    )
