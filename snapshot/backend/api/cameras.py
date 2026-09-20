"""摄像头与视频流接口。

视频流第一版使用 MJPEG（multipart/x-mixed-replace），浏览器 5~10 FPS 即可。
"""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse

from ..schemas.api import ActionResult, CameraPatch
from ..schemas.common import CameraOut
from ..services.state import AppState
from .deps import get_state

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("/{camera_id}/capture-info")
async def capture_info(camera_id: str, state: AppState = Depends(get_state)):
    worker = state.cameras.get(camera_id) if state.cameras else None
    if worker is None:
        raise HTTPException(404, '摄像头不存在')
    return dict(camera_id=camera_id, online=worker.online,
                capture=getattr(worker.source, 'capture_info', {}))


@router.get("", response_model=list[CameraOut])
async def cameras(state: AppState = Depends(get_state)) -> list[CameraOut]:
    return state.cameras.snapshots() if state.cameras else []


@router.get("/{camera_id}", response_model=CameraOut)
async def camera(camera_id: str, state: AppState = Depends(get_state)) -> CameraOut:
    worker = state.cameras.get(camera_id) if state.cameras else None
    if worker is None:
        raise HTTPException(status_code=404, detail=f"摄像头 {camera_id} 不存在")
    return worker.snapshot()


@router.patch("/{camera_id}", response_model=ActionResult)
async def patch_camera(
    camera_id: str, payload: CameraPatch, state: AppState = Depends(get_state)
) -> ActionResult:
    if state.cameras is None or state.cameras.get(camera_id) is None:
        raise HTTPException(status_code=404, detail=f"摄像头 {camera_id} 不存在")
    fields = payload.model_dump(exclude_none=True)
    state.cameras.apply_config(camera_id, **fields)
    await asyncio.to_thread(state.repo.update_camera_config, camera_id, **fields)
    state.repo.insert_operator_action(None, f"camera_config:{camera_id}", "local", str(fields))
    return ActionResult(ok=True, message="摄像头配置已更新", detail=fields)


@router.get("/{camera_id}/stream")
async def stream(
    camera_id: str,
    mode: Literal["raw", "annotated"] = "annotated",
    state: AppState = Depends(get_state),
) -> StreamingResponse:
    if state.cameras is None:
        raise HTTPException(status_code=503, detail="摄像头服务未就绪")
    generator = state.cameras.mjpeg_stream(camera_id, annotated=mode == "annotated")
    return StreamingResponse(
        generator,
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


@router.get("/{camera_id}/snapshot")
async def snapshot(
    camera_id: str,
    mode: Literal["raw", "annotated"] = "annotated",
    state: AppState = Depends(get_state),
) -> Response:
    if state.cameras is None:
        raise HTTPException(status_code=503, detail="摄像头服务未就绪")
    payload = state.cameras.snapshot_jpeg(camera_id, annotated=mode == "annotated")
    if payload is None:
        raise HTTPException(status_code=404, detail="暂无画面")
    return Response(content=payload, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@router.post("/{camera_id}/reconnect", response_model=ActionResult)
async def reconnect(camera_id: str, state: AppState = Depends(get_state)) -> ActionResult:
    worker = state.cameras.get(camera_id) if state.cameras else None
    if worker is None:
        raise HTTPException(status_code=404, detail=f"摄像头 {camera_id} 不存在")

    def _restart() -> None:
        worker.source.close()
        worker.source.open()

    await asyncio.to_thread(_restart)
    return ActionResult(ok=True, message=f"{camera_id} 已尝试重新连接")
