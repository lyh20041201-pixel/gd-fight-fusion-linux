"""历史趋势数据接口（ECharts 数据源）。"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, Query

from ..devices.manager import SENSOR_META
from ..schemas.api import TrendSeries, TrendsOut
from ..services.state import AppState
from .deps import get_state

router = APIRouter(prefix="/api/readings", tags=["readings"])

METRIC_LABELS = {
    "temperature": ("温度", "°C"),
    "humidity": ("湿度", "%"),
    "co2": ("CO₂", "ppm"),
    "light": ("光照", "lux"),
    "noise": ("噪声", "dB"),
    "occupancy": ("人数", "人"),
    "events": ("报警次数", "次"),
}


@router.get("/trends", response_model=TrendsOut)
async def trends(
    metrics: str = Query("co2,noise,occupancy", description="逗号分隔的指标"),
    start: float | None = None,
    end: float | None = None,
    bucket_seconds: int = Query(60, ge=10, le=3600),
    camera_id: str | None = None,
    state: AppState = Depends(get_state),
) -> TrendsOut:
    end = end or time.time()
    start = start or (end - 3600 * 6)
    wanted = [m.strip() for m in metrics.split(",") if m.strip()]
    series: list[TrendSeries] = []

    for metric in wanted:
        label, unit = METRIC_LABELS.get(metric, (metric, ""))
        if metric == "occupancy":
            points = await asyncio.to_thread(
                state.repo.occupancy_series, start, end, bucket_seconds, camera_id
            )
        elif metric == "events":
            points = await asyncio.to_thread(
                state.repo.event_count_series, start, end, bucket_seconds
            )
        elif metric in SENSOR_META:
            points = await asyncio.to_thread(
                state.repo.reading_series, metric, start, end, bucket_seconds
            )
        else:
            continue
        series.append(TrendSeries(metric=metric, label=label, unit=unit, points=points))

    return TrendsOut(start=start, end=end, bucket_seconds=bucket_seconds, series=series)


@router.get("/availability")
async def availability(
    hours: int = Query(24, ge=1, le=168), state: AppState = Depends(get_state)
) -> dict[str, float]:
    """节点与摄像头在线率（按当前状态与最近事件粗略统计）。"""
    devices = state.devices.devices() if state.devices else []
    cameras = state.cameras.snapshots() if state.cameras else []
    node_rate = (
        sum(1 for d in devices if d.online) / len(devices) if devices else 0.0
    )
    camera_rate = (
        sum(1 for c in cameras if c.online) / len(cameras) if cameras else 0.0
    )
    return {
        "node_online_rate": round(node_rate, 3),
        "camera_online_rate": round(camera_rate, 3),
        "window_hours": float(hours),
    }
