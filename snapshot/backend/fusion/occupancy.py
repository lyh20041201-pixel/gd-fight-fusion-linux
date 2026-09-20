"""多摄像头区域融合。

第一版策略（不做跨摄像头 ReID / 人脸识别）：
- 每路摄像头负责一个教室区域，只统计落在自身有效 ROI 内的人员；
- 同一区域存在多路摄像头时，指定其中一路为主摄像头参与全局求和；
- 全局人数 = 各主摄像头有效区域人数之和，并对最近数秒做中位数平滑；
- 同一区域主/副摄像头人数长期差异过大 -> 上报"多摄像头人数矛盾"。
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CameraOccupancy:
    camera_id: str
    region: str
    is_primary: bool
    count: int = 0
    online: bool = True
    updated_at: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=120))


class OccupancyFusion:
    def __init__(self, median_window_seconds: float = 3.0, spike_window_seconds: float = 10.0):
        self.median_window_seconds = median_window_seconds
        self.spike_window_seconds = spike_window_seconds
        self._lock = threading.RLock()
        self._cameras: dict[str, CameraOccupancy] = {}
        self._global_history: deque[tuple[float, int]] = deque(maxlen=600)
        self._started_at = time.time()

    # ---------- 注册与更新 ----------
    def register(self, camera_id: str, region: str, is_primary: bool) -> None:
        with self._lock:
            existing = self._cameras.get(camera_id)
            if existing is None:
                self._cameras[camera_id] = CameraOccupancy(camera_id, region, is_primary)
            else:
                existing.region = region
                existing.is_primary = is_primary

    def unregister(self, camera_id: str) -> None:
        with self._lock:
            self._cameras.pop(camera_id, None)

    def update(self, camera_id: str, count: int, online: bool, timestamp: float | None = None) -> None:
        ts = timestamp or time.time()
        with self._lock:
            cam = self._cameras.get(camera_id)
            if cam is None:
                cam = CameraOccupancy(camera_id, "", True)
                self._cameras[camera_id] = cam
            cam.count = max(0, int(count))
            cam.online = online
            cam.updated_at = ts
            cam.history.append((ts, cam.count))
            self._global_history.append((ts, self._raw_total_locked()))

    # ---------- 查询 ----------
    def _raw_total_locked(self) -> int:
        selected: dict[str, CameraOccupancy] = {}
        for cam in self._cameras.values():
            if not cam.online:
                continue
            key = cam.region or cam.camera_id
            previous = selected.get(key)
            # 每个区域只选一路；主机优先，同优先级按 ID 确定，消除注册顺序影响。
            if previous is None or (not cam.is_primary, cam.camera_id) < (
                not previous.is_primary, previous.camera_id
            ):
                selected[key] = cam
        return sum(cam.count for cam in selected.values())

    def raw_total(self) -> int:
        with self._lock:
            return self._raw_total_locked()

    def smoothed_total(self) -> int:
        """最近 median_window_seconds 内的中位数，抑制单帧抖动。"""
        cutoff = time.time() - self.median_window_seconds
        with self._lock:
            values = [v for ts, v in self._global_history if ts >= cutoff]
            if not values:
                return self._raw_total_locked()
            return int(round(statistics.median(values)))

    def per_camera(self) -> dict[str, int]:
        with self._lock:
            return {c.camera_id: c.count for c in self._cameras.values()}

    def trend(self) -> str:
        cutoff = time.time() - self.spike_window_seconds
        with self._lock:
            values = [v for ts, v in self._global_history if ts >= cutoff]
        if len(values) < 4:
            return "stable"
        head = statistics.median(values[: max(1, len(values) // 3)])
        tail = statistics.median(values[-max(1, len(values) // 3):])
        if tail > head + 1:
            return "rising"
        if tail < head - 1:
            return "falling"
        return "stable"

    def spike_delta(self) -> int:
        """最近窗口内最大值与最小值之差，用于"人数突然大幅变化"。

        刚启动时历史不足一个完整窗口，直接返回 0，避免把"从 0 到正常人数"
        的冷启动过程误判为人数突变。
        """
        now = time.time()
        # 冷启动阶段（不足两个窗口）人数从 0 爬升到正常值，不能算突变
        if now - self._started_at < self.spike_window_seconds * 2:
            return 0
        cutoff = now - self.spike_window_seconds
        with self._lock:
            window = [(ts, v) for ts, v in self._global_history if ts >= cutoff]
        if len(window) < 3:
            return 0
        values = [v for _ts, v in window]
        return max(values) - min(values)

    def region_conflicts(self, delta_threshold: int) -> list[dict[str, Any]]:
        """同区域多摄像头人数矛盾。"""
        conflicts: list[dict[str, Any]] = []
        with self._lock:
            by_region: dict[str, list[CameraOccupancy]] = {}
            for cam in self._cameras.values():
                if not cam.online or not cam.region:
                    continue
                by_region.setdefault(cam.region, []).append(cam)
            for region, cams in by_region.items():
                if len(cams) < 2:
                    continue
                counts = [c.count for c in cams]
                gap = max(counts) - min(counts)
                if gap >= delta_threshold:
                    primary = next((c for c in cams if c.is_primary), cams[0])
                    conflicts.append(
                        {
                            "region": region,
                            "delta": gap,
                            "primary_camera": primary.camera_id,
                            "counts": {c.camera_id: c.count for c in cams},
                        }
                    )
        return conflicts

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            cams = {
                c.camera_id: {
                    "count": c.count,
                    "region": c.region,
                    "is_primary": c.is_primary,
                    "online": c.online,
                    "updated_at": c.updated_at,
                }
                for c in self._cameras.values()
            }
        return {
            "total": self.smoothed_total(),
            "raw_total": self.raw_total(),
            "trend": self.trend(),
            "cameras": cams,
        }
