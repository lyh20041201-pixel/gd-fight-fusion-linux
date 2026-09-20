"""风险事件接口。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse

from ..schemas.events import EventListOut, EventPatch, RiskEventOut
from ..services.state import AppState
from ..services.event_media import MediaDeleteError, delete_event_with_media
from .deps import get_state

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("", response_model=EventListOut)
async def list_events(
    limit: int = Query(30, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: str | None = None,
    risk_level: str | None = None,
    event_type: str | None = None,
    camera_id: str | None = None,
    start: float | None = None,
    end: float | None = None,
    state: AppState = Depends(get_state),
) -> EventListOut:
    total, items = await asyncio.to_thread(
        state.repo.list_events,
        limit=limit,
        offset=offset,
        status=status,
        risk_level=risk_level,
        event_type=event_type,
        camera_id=camera_id,
        start=start,
        end=end,
    )
    return EventListOut(total=total, items=items)


@router.get("/images/{path:path}")
async def event_image(path: str, state: AppState = Depends(get_state)) -> FileResponse:
    """按相对路径返回事件图片（限制在事件目录内，防止路径穿越）。"""
    root = state.settings.events_dir.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_file() or target.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(Path(target), media_type='image/png' if target.suffix.lower() == '.png' else 'image/jpeg')


@router.get('/{event_id}/audio')
async def event_audio(event_id: str, state: AppState = Depends(get_state)) -> FileResponse:
    event = await asyncio.to_thread(state.repo.get_event, event_id)
    relative = (event.rule_basis.get('audio') or {}).get('path') if event else None
    if not isinstance(relative, str):
        raise HTTPException(status_code=404, detail='此事件没有录音')
    root = state.settings.events_dir.resolve()
    target = (root / relative).resolve()
    if (not target.is_relative_to(root) or target.parent.name != event_id
            or target.name != 'audio.wav' or not target.is_file()):
        raise HTTPException(status_code=404, detail='录音不存在')
    return FileResponse(target, media_type='audio/wav', headers={'Cache-Control': 'private, no-store'})


@router.get('/{event_id}/clip')
async def event_clip(event_id: str, state: AppState = Depends(get_state)) -> FileResponse:
    event = await asyncio.to_thread(state.repo.get_event, event_id)
    relative = (event.rule_basis.get('visual') or {}).get('evidence_clip') if event else None
    if not isinstance(relative, str):
        raise HTTPException(status_code=404, detail='此事件没有证据视频')
    root = state.settings.events_dir.resolve()
    target = (root / relative).resolve()
    if (not target.is_relative_to(root) or target.parent.name != event_id
            or target.name != 'evidence.avi' or not target.is_file()):
        raise HTTPException(status_code=404, detail='证据视频不存在')
    return FileResponse(target, media_type='video/x-msvideo', filename=f'{event_id}.avi',
                        headers={'Cache-Control': 'private, no-store'})


@router.get("/{event_id}", response_model=RiskEventOut)
async def get_event(event_id: str, state: AppState = Depends(get_state)) -> RiskEventOut:
    event = await asyncio.to_thread(state.repo.get_event, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="事件不存在")
    return event


@router.delete('/{event_id}', status_code=204)
async def delete_event(event_id: str, state: AppState = Depends(get_state)) -> Response:
    saving = any(not task.done() and task.get_name() == f'event-{event_id}'
                 for task in getattr(state.events, '_tasks', ()))
    reviewing = event_id in getattr(state.qwen_queue, 'pending_events', ())
    if saving or reviewing:
        raise HTTPException(409, '事件正在保存或复核，请完成后再删除')
    try:
        review_lock = getattr(state.visual_events, '_lock', None) or asyncio.Lock()
        async with review_lock:
            await asyncio.to_thread(delete_event_with_media, state.repo, state.settings.events_dir, event_id)
    except MediaDeleteError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    pending = await asyncio.to_thread(state.repo.count_events, status='PENDING')
    state.ws.broadcast('risk_event_deleted', {'event_id': event_id, 'pending_events': pending, 'active_events': pending})
    return Response(status_code=204)


@router.patch("/{event_id}", response_model=RiskEventOut)
async def patch_event(
    event_id: str, payload: EventPatch, state: AppState = Depends(get_state)
) -> RiskEventOut:
    if state.events is None:
        raise HTTPException(status_code=503, detail="事件服务未就绪")
    updated = await state.events.patch_event(
        event_id,
        status=payload.status.value if payload.status else None,
        note=payload.note,
        operator=payload.operator,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="事件不存在")
    return updated
