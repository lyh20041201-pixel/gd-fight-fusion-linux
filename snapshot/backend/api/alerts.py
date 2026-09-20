"""报警控制接口。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ..schemas.api import ActionResult, AlertAckRequest, AlertMuteRequest, AlertTestRequest
from ..schemas.common import AlertState
from ..services.state import AppState
from .deps import get_state

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


def _controller(state: AppState):
    if state.alerts is None:
        raise HTTPException(status_code=503, detail="报警控制器未就绪")
    return state.alerts


@router.get("", response_model=AlertState)
async def alert_state(state: AppState = Depends(get_state)) -> AlertState:
    return state.alert_state()


@router.get("/actions")
async def alert_actions(
    limit: int = Query(50, ge=1, le=500), state: AppState = Depends(get_state)
) -> list[dict[str, Any]]:
    return _controller(state).recent_actions(limit)


@router.post("/test", response_model=ActionResult)
async def test_alert(
    payload: AlertTestRequest, state: AppState = Depends(get_state)
) -> ActionResult:
    """测试 RGB 灯 / 蜂鸣器。真实模式必须携带 confirm=true（前端二次确认）。"""
    result = await _controller(state).test(
        target=payload.target,
        # 传枚举的字面值，保持下发指令与操作日志的原有格式
        color=payload.color.value,
        buzzer_mode=payload.buzzer_mode.value,
        duration_seconds=payload.duration_seconds,
        confirm=payload.confirm,
    )
    return ActionResult(
        ok=bool(result.get("ok")),
        message=str(result.get("message", "测试指令已下发")),
        detail={k: v for k, v in result.items() if k not in {"ok", "message"}},
    )


@router.post("/mute", response_model=ActionResult)
async def mute(payload: AlertMuteRequest, state: AppState = Depends(get_state)) -> ActionResult:
    result = await _controller(state).mute(payload.seconds, payload.operator)
    return ActionResult(ok=True, message="已临时静音（红灯与报警状态保留）", detail=result)


@router.post("/unmute", response_model=ActionResult)
async def unmute(payload: AlertAckRequest, state: AppState = Depends(get_state)) -> ActionResult:
    result = await _controller(state).unmute(payload.operator)
    return ActionResult(ok=True, message="已取消静音", detail=result)


@router.post("/acknowledge", response_model=ActionResult)
async def acknowledge(
    payload: AlertAckRequest, state: AppState = Depends(get_state)
) -> ActionResult:
    result = await _controller(state).acknowledge(payload.operator, payload.note)
    return ActionResult(ok=True, message="报警已确认", detail=result)


@router.post("/clear", response_model=ActionResult)
async def clear(payload: AlertAckRequest, state: AppState = Depends(get_state)) -> ActionResult:
    result = await _controller(state).clear(payload.operator)
    return ActionResult(ok=True, message="报警已人工解除", detail=result)
