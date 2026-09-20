"""系统设置接口。

注意：Qwen 密钥只从环境变量读取，此处仅返回"是否已配置"，绝不返回密钥内容，
也不接受前端写入密钥。
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends

from ..cameras.usb_source import probe_cameras
from ..devices.serial_gateway import list_serial_ports
from ..schemas.api import SettingsOut, SettingsPatch
from ..services.state import AppState
from .deps import get_state

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/settings", tags=["settings"])


def _build(state: AppState) -> SettingsOut:
    runtime = state.runtime
    cameras = [
        {
            "camera_id": c.camera_id,
            "name": c.name,
            "region": c.region,
            "source": c.source,
            "roi": c.roi,
            "is_primary_overlap": c.is_primary_overlap,
            "online": c.online,
        }
        for c in (state.cameras.snapshots() if state.cameras else [])
    ]
    return SettingsOut(
        simulation_mode=runtime.simulation_mode,
        serial_port=str(runtime.get("serial_port", "") or ""),
        available_serial_ports=list_serial_ports(),
        camera_device_indices=list(runtime.get("camera_device_indices") or []),
        available_cameras=cameras,
        qwen_enabled=runtime.qwen_enabled,
        qwen_model=runtime.qwen_model,
        qwen_configured=state.settings.qwen_configured,
        audio_enabled=bool(runtime.get('audio_enabled', False)),
        audio_device_index=int(runtime.get('audio_device_index', -1)),
        audio_status=state.audio.status(),
        face_blur_enabled=runtime.face_blur_enabled,
        event_image_dir=str(state.settings.events_dir),
        data_retention_days=runtime.data_retention_days,
        thresholds=runtime.thresholds,
        schedule=runtime.schedule,
        restart_required=runtime.restart_required,
    )


@router.get("", response_model=SettingsOut)
async def get_settings(state: AppState = Depends(get_state)) -> SettingsOut:
    return _build(state)


@router.put("", response_model=SettingsOut)
async def update_settings(
    payload: SettingsPatch, state: AppState = Depends(get_state)
) -> SettingsOut:
    patch = payload.model_dump(exclude_none=True)
    previous_audio = {key: state.runtime.get(key) for key in ('audio_enabled', 'audio_device_index')}
    applied = await asyncio.to_thread(state.runtime.update, patch)
    if any(key in applied and applied[key] != value for key, value in previous_audio.items()) or applied.get('simulation_mode') is True:
        async with state._audio_config_lock:
            await asyncio.to_thread(state.audio.stop)
            state.audio.start()
    if "camera_device_indices" in applied:
        # 立即按新选择重建摄像头（真实模式下无需重启）
        result = await state.apply_camera_selection(list(applied["camera_device_indices"]))
        if not result.get("ok"):
            logger.warning("摄像头切换未生效: %s", result.get("message"))
    if "face_blur_enabled" in applied and state.cameras:
        state.cameras.set_face_blur(bool(applied["face_blur_enabled"]))
    if "thresholds" in applied and state.rules:
        # 阈值改变后同步到相关规则参数
        thresholds = applied["thresholds"]
        state.rules.update_rule(
            "co2_warning",
            params={
                "enter": thresholds.get("co2_warning"),
                "clear": thresholds.get("co2_warning_clear"),
                "duration_seconds": thresholds.get("co2_duration_seconds"),
            },
        )
        state.rules.update_rule(
            "co2_critical",
            params={
                "enter": thresholds.get("co2_critical"),
                "clear": thresholds.get("co2_critical_clear"),
                "duration_seconds": thresholds.get("co2_duration_seconds"),
            },
        )
        state.rules.update_rule(
            "noise_abnormal",
            params={
                "enter": thresholds.get("noise_warning"),
                "clear": thresholds.get("noise_clear"),
                "duration_seconds": thresholds.get("noise_duration_seconds"),
            },
        )
        state.rules.update_rule(
            "over_capacity", params={"limit": thresholds.get("occupancy_limit")}
        )
    state.repo.insert_operator_action(None, "settings_update", "local", str(list(applied)))
    return _build(state)


@router.get("/probe-cameras")
async def probe(max_index: int = 6, state: AppState = Depends(get_state)) -> list[dict[str, object]]:
    """探测本机 USB 摄像头（真实模式配置用）。"""
    return await asyncio.to_thread(probe_cameras, max_index, state.settings.camera_backend)


@router.get('/audio-devices')
async def audio_devices(state: AppState = Depends(get_state)):
    return await asyncio.to_thread(state.audio.devices)
