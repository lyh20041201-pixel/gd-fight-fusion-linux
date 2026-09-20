"""有界帧环形缓冲。

- 保存最近约 N 秒的关键帧（默认 5s，按抽帧频率存储，避免内存暴涨）。
- 只保留最新帧的槽位（LatestFrameSlot）用于推流与推理，处理不过来时丢弃旧帧。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import numpy as np


class LatestFrameSlot:
    """单槽有界缓冲：写入总是覆盖，读取取最新。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: tuple[np.ndarray, float, dict[str, Any]] | None = None
        self.dropped = 0

    def put(self, image: np.ndarray, timestamp: float, meta: dict[str, Any] | None = None) -> None:
        with self._lock:
            if self._value is not None:
                self.dropped += 1
            self._value = (image, timestamp, meta or {})

    def get(self) -> tuple[np.ndarray, float, dict[str, Any]] | None:
        with self._lock:
            return self._value

    def take(self) -> tuple[np.ndarray, float, dict[str, Any]] | None:
        with self._lock:
            value, self._value = self._value, None
            return value


class FrameRingBuffer:
    """最近 N 秒关键帧缓冲，用于事件前后帧回溯。"""

    def __init__(self, seconds: float = 5.0, sample_interval: float = 0.5) -> None:
        self.seconds = seconds
        self.sample_interval = sample_interval
        self._buffer: deque[tuple[float, np.ndarray, dict[str, Any]]] = deque(
            maxlen=max(4, int(seconds / sample_interval) + 2)
        )
        self._lock = threading.Lock()
        self._last_sample = 0.0

    def offer(self, image: np.ndarray, timestamp: float, meta: dict[str, Any] | None = None) -> None:
        if timestamp - self._last_sample < self.sample_interval:
            return
        self._last_sample = timestamp
        with self._lock:
            self._buffer.append((timestamp, image.copy(), meta or {}))

    def snapshot(self) -> list[tuple[float, np.ndarray, dict[str, Any]]]:
        with self._lock:
            return list(self._buffer)

    def attach_evidence(self, timestamp, annotated, evidence_raw):
        """Attach only to the exact captured frame, never a later detector result."""
        with self._lock:
            for i in range(len(self._buffer) - 1, -1, -1):
                ts, image, meta = self._buffer[i]
                if ts == timestamp:
                    self._buffer[i] = (ts, image, {**meta, 'annotated': annotated.copy(), 'evidence_raw': evidence_raw.copy()})
                    return

    def frame_before(self, reference: float, offset: float) -> tuple[float, np.ndarray, dict[str, Any]] | None:
        """取 reference-offset 附近最接近的一帧。"""
        target = reference - offset
        best: tuple[float, np.ndarray, dict[str, Any]] | None = None
        best_delta = float("inf")
        for ts, image, meta in self.snapshot():
            delta = abs(ts - target)
            if delta < best_delta:
                best, best_delta = (ts, image, meta), delta
        return best

    def newest(self) -> tuple[float, np.ndarray, dict[str, Any]] | None:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def span_seconds(self) -> float:
        with self._lock:
            if len(self._buffer) < 2:
                return 0.0
            return self._buffer[-1][0] - self._buffer[0][0]


def now() -> float:
    return time.time()
