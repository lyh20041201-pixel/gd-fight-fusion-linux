"""REST 接口的请求 / 响应模型。

请求模型一律继承 StrictModel：拒绝未知字段，并对所有可写字段设定取值范围。
这些接口直接改写控制循环使用的参数，非法值不会当场报错，而是让规则永远无法
触发 / 永远无法解除，或让后台线程反复抛异常。
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..config.defaults import (
    MAX_RESTRICTED_PERIODS,
    RULE_PARAM_BOUNDS,
    SCHEDULE_KEYS,
    THRESHOLD_BOUNDS,
)
from .enums import BuzzerMode, LedColor, RiskLevel

_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class StrictModel(BaseModel):
    """请求体基类：未知字段直接 422，而不是被静默忽略。"""

    model_config = ConfigDict(extra="forbid")


def _check_number(name: str, value: Any, bounds: tuple[float, float]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数字")
    low, high = bounds
    if not low <= float(value) <= high:
        raise ValueError(f"{name} 超出允许范围 [{low}, {high}]：{value}")
    return float(value)


class RuleOut(BaseModel):
    rule_id: str
    name: str
    description: str = ""
    enabled: bool = True
    event_type: str
    risk_level: str
    params: dict[str, Any] = Field(default_factory=dict)
    editable_params: list[str] = Field(default_factory=list)
    last_triggered_at: float | None = None
    state: str = "idle"


class RulePatch(StrictModel):
    enabled: bool | None = None
    # 非法等级会让规则引擎在 RiskLevel(...) 处持续抛异常
    risk_level: RiskLevel | None = None
    params: dict[str, Any] | None = None

    @field_validator("params")
    @classmethod
    def _check_params(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return value
        for key, raw in value.items():
            bounds = RULE_PARAM_BOUNDS.get(key)
            if bounds is None:
                raise ValueError(f"未知或不可修改的规则参数：{key}")
            _check_number(f"规则参数 {key}", raw, bounds)
        return value


class CameraPatch(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=64)
    region: str | None = Field(default=None, min_length=1, max_length=64)
    roi: list[list[float]] | None = None
    is_primary_overlap: bool | None = None
    enabled: bool | None = None

    @field_validator("roi")
    @classmethod
    def _check_roi(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return value
        if not 3 <= len(value) <= 20:
            raise ValueError("ROI 至少 3 个点、至多 20 个点")
        for point in value:
            if len(point) != 2:
                raise ValueError("ROI 的每个点必须是 [x, y]")
            if not all(0.0 <= float(c) <= 1.0 for c in point):
                raise ValueError("ROI 坐标必须是 0~1 的归一化值")
        return value


class SettingsOut(BaseModel):
    simulation_mode: bool
    serial_port: str
    available_serial_ports: list[dict[str, str]] = Field(default_factory=list)
    camera_device_indices: list[int] = Field(default_factory=list)
    available_cameras: list[dict[str, Any]] = Field(default_factory=list)
    qwen_enabled: bool
    qwen_model: str
    qwen_configured: bool
    audio_enabled: bool = True
    audio_device_index: int = -1
    audio_status: dict[str, Any] = Field(default_factory=dict)
    face_blur_enabled: bool
    event_image_dir: str
    data_retention_days: int
    thresholds: dict[str, Any] = Field(default_factory=dict)
    schedule: dict[str, Any] = Field(default_factory=dict)
    restart_required: bool = False


class SettingsPatch(StrictModel):
    simulation_mode: bool | None = None
    serial_port: str | None = Field(default=None, max_length=128)
    camera_device_indices: list[int] | None = Field(default=None, max_length=8)
    qwen_enabled: bool | None = None
    qwen_model: Literal['qwen3.5-omni-flash', 'qwen3.5-omni-plus'] | None = None
    audio_enabled: bool | None = None
    audio_device_index: int | None = Field(default=None, ge=-1, le=1024)
    face_blur_enabled: bool | None = None
    event_image_dir: str | None = Field(default=None, min_length=1, max_length=256)
    # 负数会让清理游标落到未来，等于清空历史表；下限 1 天、上限 10 年。
    data_retention_days: int | None = Field(default=None, ge=1, le=3650)
    thresholds: dict[str, Any] | None = None
    schedule: dict[str, Any] | None = None

    @field_validator("camera_device_indices")
    @classmethod
    def _check_indices(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return value
        for index in value:
            if not 0 <= index <= 64:
                raise ValueError(f"摄像头设备号超出范围：{index}")
        return value

    @field_validator("thresholds")
    @classmethod
    def _check_thresholds(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return value
        for key, raw in value.items():
            bounds = THRESHOLD_BOUNDS.get(key)
            if bounds is None:
                raise ValueError(f"未知阈值项：{key}")
            _check_number(f"阈值 {key}", raw, bounds)
        return value

    @field_validator("schedule")
    @classmethod
    def _check_schedule(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return value
        unknown = sorted(set(value) - SCHEDULE_KEYS)
        if unknown:
            raise ValueError(f"未知作息项：{unknown}")
        for key in ("class_start", "class_end"):
            if key in value and not _HHMM.match(str(value[key])):
                raise ValueError(f"{key} 必须是 HH:MM 格式")
        if "after_hours_duration_seconds" in value:
            _check_number(
                "after_hours_duration_seconds",
                value["after_hours_duration_seconds"],
                (0.0, 86400.0),
            )
        periods = value.get("restricted_periods")
        if periods is not None:
            if not isinstance(periods, list) or len(periods) > MAX_RESTRICTED_PERIODS:
                raise ValueError(
                    f"restricted_periods 必须是不超过 {MAX_RESTRICTED_PERIODS} 项的数组"
                )
            for period in periods:
                # 非字典元素会让规则引擎在 p.get(...) 处崩掉整个控制循环
                if not isinstance(period, dict) or set(period) - {"start", "end"}:
                    raise ValueError("restricted_periods 的每项必须是 start / end 两个键")
                for key in ("start", "end"):
                    if not _HHMM.match(str(period.get(key, ""))):
                        raise ValueError(f"restricted_periods 的 {key} 必须是 HH:MM 格式")
        return value


class AlertTestRequest(StrictModel):
    target: Literal["led", "buzzer", "both"] = "led"
    # 非法颜色 / 蜂鸣模式会让控制器在枚举转换处抛异常
    color: LedColor = LedColor.RED
    buzzer_mode: BuzzerMode = BuzzerMode.PULSE
    duration_seconds: float = Field(default=3.0, gt=0, le=60)
    confirm: bool = False  # 真实模式下必须为 true（前端二次确认）


class AlertMuteRequest(StrictModel):
    # 上限 1 小时：不允许把报警无限期静音
    seconds: float = Field(default=300.0, gt=0, le=3600)
    operator: str = Field(default="local", min_length=1, max_length=64)


class AlertAckRequest(StrictModel):
    operator: str = Field(default="local", min_length=1, max_length=64)
    note: str | None = Field(default=None, max_length=1000)


class SimulateRequest(StrictModel):
    scenario: Literal[
        "after_hours_presence",
        "over_capacity",
        "smoke_detected",
        "co2_critical",
        "noise_abnormal",
        "camera_obstructed",
        "node_offline",
        "camera_offline",
        "recover",
    ] = "after_hours_presence"
    node_id: str | None = Field(default=None, min_length=1, max_length=64)
    camera_id: str | None = Field(default=None, min_length=1, max_length=64)


class TrendSeries(BaseModel):
    metric: str
    label: str
    unit: str = ""
    points: list[list[float]] = Field(default_factory=list)  # [[ts, value], ...]


class TrendsOut(BaseModel):
    start: float
    end: float
    bucket_seconds: int
    series: list[TrendSeries]


class ActionResult(BaseModel):
    ok: bool
    message: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)
