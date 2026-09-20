"""设备管理接口。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..devices.serial_gateway import list_serial_ports
from ..schemas.api import ActionResult
from ..schemas.common import DeviceOut, NodeStats
from ..services.state import AppState
from .deps import get_state

router = APIRouter(prefix="/api/devices", tags=["devices"])


@router.get("", response_model=list[DeviceOut])
async def devices(state: AppState = Depends(get_state)) -> list[DeviceOut]:
    if state.devices is None:
        return []
    return state.devices.devices()


@router.get("/stats", response_model=list[NodeStats])
async def device_stats(state: AppState = Depends(get_state)) -> list[NodeStats]:
    if state.devices is None:
        return []
    return state.devices.node_stats()


@router.get("/serial-ports")
async def serial_ports() -> list[dict[str, str]]:
    return list_serial_ports()


@router.post("/{device_id}/reconnect", response_model=ActionResult)
async def reconnect(device_id: str, state: AppState = Depends(get_state)) -> ActionResult:
    if state.devices is None:
        raise HTTPException(status_code=503, detail="设备管理器未就绪")
    result: dict[str, Any] = await state.devices.reconnect(device_id)
    state.repo.insert_operator_action(None, f"reconnect:{device_id}", "local", None)
    return ActionResult(
        ok=bool(result.get("ok")),
        message=str(result.get("message", "")),
        detail=result.get("detail", {}),
    )
