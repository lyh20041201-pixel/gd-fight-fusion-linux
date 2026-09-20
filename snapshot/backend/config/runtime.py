"""运行时可修改配置。

分层：环境变量(.env) -> SQLite settings 表覆盖 -> 内存缓存。
系统设置页修改的是 settings 表，重启后依旧生效；密钥永远只来自环境变量。
"""

from __future__ import annotations

import copy
import logging
import threading
from typing import Any

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..database.repositories import Repository
from .defaults import DEFAULT_SCHEDULE, DEFAULT_THRESHOLDS
from .settings import Settings

logger = logging.getLogger(__name__)

_OVERRIDABLE = {
    "simulation_mode": bool,
    "serial_port": str,
    "camera_device_indices": list,
    "qwen_enabled": bool,
    "qwen_model": str,
    "audio_enabled": bool,
    "audio_device_index": int,
    "face_blur_enabled": bool,
    "event_image_dir": str,
    "data_retention_days": int,
}


class RuntimeConfig:
    @property
    def settings(self) -> Settings:
        return self._settings

    def __init__(self, settings: Settings, repo: Repository) -> None:
        self._settings = settings
        self._repo = repo
        self._lock = threading.RLock()
        self._cache: dict[str, Any] = {}
        self.restart_required = False

    # ---------- 加载 ----------
    def load(self) -> None:
        with self._lock:
            stored = self._repo.all_settings()
            thresholds = copy.deepcopy(DEFAULT_THRESHOLDS)
            thresholds.update(stored.get("thresholds") or {})
            schedule = copy.deepcopy(DEFAULT_SCHEDULE)
            schedule.update(stored.get("schedule") or {})

            self._cache = {
                "thresholds": thresholds,
                "schedule": schedule,
                "simulation_mode": stored.get("simulation_mode", self._settings.simulation_mode),
                "serial_port": stored.get("serial_port", self._settings.serial_port),
                "camera_device_indices": stored.get(
                    "camera_device_indices", self._settings.camera_indices
                ),
                "qwen_enabled": stored.get("qwen_enabled", self._settings.qwen_enabled),
                "qwen_model": stored.get("qwen_model", self._settings.qwen_model),
                "audio_enabled": stored.get("audio_enabled", self._settings.audio_enabled),
                "audio_device_index": stored.get("audio_device_index", self._settings.audio_device_index),
                "face_blur_enabled": stored.get(
                    "face_blur_enabled", self._settings.face_blur_enabled
                ),
                "event_image_dir": stored.get(
                    "event_image_dir", str(self._settings.events_dir)
                ),
                "data_retention_days": stored.get(
                    "data_retention_days", self._settings.data_retention_days
                ),
            }

    # ---------- 读 ----------
    @property
    def thresholds(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._cache.get("thresholds", DEFAULT_THRESHOLDS))

    @property
    def schedule(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._cache.get("schedule", DEFAULT_SCHEDULE))

    def threshold(self, key: str, default: Any = None) -> Any:
        return self.thresholds.get(key, DEFAULT_THRESHOLDS.get(key, default))

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._cache.get(key, default)

    @property
    def simulation_mode(self) -> bool:
        return bool(self.get("simulation_mode", True))

    @property
    def qwen_enabled(self) -> bool:
        return bool(self.get("qwen_enabled", False))

    @property
    def qwen_model(self) -> str:
        value = self.get("qwen_model", self._settings.qwen_model)
        return value if value in {'qwen3.5-omni-flash', 'qwen3.5-omni-plus'} else self._settings.qwen_model

    @property
    def face_blur_enabled(self) -> bool:
        return bool(self.get("face_blur_enabled", True))

    @property
    def data_retention_days(self) -> int:
        return max(1, int(self.get("data_retention_days", 30)))

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._cache)

    # ---------- 写 ----------
    def update(self, patch: dict[str, Any]) -> dict[str, Any]:
        """写入 settings 表并刷新缓存，返回实际生效的变更。"""
        applied: dict[str, Any] = {}
        with self._lock:
            for key, value in patch.items():
                if value is None:
                    continue
                if key == 'qwen_model' and value not in {'qwen3.5-omni-flash', 'qwen3.5-omni-plus'}:
                    raise ValueError('仅支持 Qwen3.5-Omni Flash / Plus')
                if key in ("thresholds", "schedule"):
                    merged = dict(self._cache.get(key, {}))
                    merged.update(value)
                    self._cache[key] = merged
                    self._repo.set_setting(key, merged)
                    applied[key] = merged
                elif key in _OVERRIDABLE:
                    self._cache[key] = value
                    self._repo.set_setting(key, value)
                    applied[key] = value
                    # 摄像头选择可在运行时重建，无需重启；模式与串口仍需重启
                    if key in {"simulation_mode", "serial_port"}:
                        self.restart_required = True
                else:
                    logger.warning("忽略未知设置项: %s", key)
        return applied
