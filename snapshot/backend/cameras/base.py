"""摄像头源抽象。

真实 USB 摄像头与模拟摄像头实现同一接口，上层（视觉流水线、MJPEG 推流、
事件抓帧）完全不区分二者。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float
    index: int
    meta: dict[str, Any] = field(default_factory=dict)


class CameraSource(abc.ABC):
    def __init__(self, camera_id: str, name: str, source: str) -> None:
        self.camera_id = camera_id
        self.name = name
        self.source = source
        self._opened = False
        self.last_error: str | None = None

    @property
    def opened(self) -> bool:
        return self._opened

    @abc.abstractmethod
    def open(self) -> bool: ...

    @abc.abstractmethod
    def read(self) -> Frame | None: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    @property
    def simulated(self) -> bool:
        return False

    def describe(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "source": self.source,
            "opened": self._opened,
            "simulated": self.simulated,
            "last_error": self.last_error,
        }
