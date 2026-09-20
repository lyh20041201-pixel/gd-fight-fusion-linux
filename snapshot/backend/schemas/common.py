"""通用数据模型：模块状态、设备、传感器、摄像头、系统状态。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .enums import (
    BuzzerMode,
    HealthState,
    LedColor,
    ModuleState,
    ModuleType,
    SafetyState,
)


class ModuleStatus(BaseModel):
    """界面「全部模块实时状态」面板的统一单元。"""

    module_id: str
    name: str
    module_type: ModuleType
    state: ModuleState = ModuleState.INITIALIZING
    value_text: str = ""  # 当前数值或运行信息
    detail: dict[str, Any] = Field(default_factory=dict)
    last_update: float | None = None  # unix 秒
    rssi: int | None = None
    packet_rate: float | None = None  # 数据包接收率 0~1
    fault_reason: str | None = None
    actions: list[str] = Field(default_factory=list)  # reconnect / test_led / test_buzzer


class SensorReadingOut(BaseModel):
    node_id: str
    sensor_kind: str
    value: float
    unit: str = ""
    timestamp: float


class NodeStats(BaseModel):
    node_id: str
    name: str
    online: bool = False
    rssi: int | None = None
    last_seen: float | None = None
    last_seq: int | None = None
    received: int = 0
    lost: int = 0
    duplicated: int = 0
    packet_rate: float = 1.0
    values: dict[str, float] = Field(default_factory=dict)


class DeviceOut(BaseModel):
    device_id: str
    name: str
    device_type: ModuleType
    state: ModuleState
    online: bool
    rssi: int | None = None
    packet_rate: float | None = None
    last_seen: float | None = None
    last_heartbeat: float | None = None
    values: dict[str, float] = Field(default_factory=dict)
    fault_reason: str | None = None
    actions: list[str] = Field(default_factory=list)


class TrackBox(BaseModel):
    track_id: int
    label: str  # 匿名编号，如 ID-12
    bbox: list[float]  # [x1, y1, x2, y2] 像素
    confidence: float
    in_roi: bool = True
    risk: bool = False
    age: int = 0


class CameraOut(BaseModel):
    camera_id: str
    name: str
    region: str = ""
    source: str = ""
    online: bool = False
    state: ModuleState = ModuleState.INITIALIZING
    person_count: int = 0
    fps: float = 0.0
    inference_ms: float = 0.0
    last_frame_at: float | None = None
    roi: list[list[float]] = Field(default_factory=list)  # 归一化多边形点
    is_primary_overlap: bool = True
    fault_reason: str | None = None
    tracks: list[TrackBox] = Field(default_factory=list)
    obstructed: bool = False


class OccupancyUpdate(BaseModel):
    total: int
    per_camera: dict[str, int]
    timestamp: float
    trend: str = "stable"  # rising / falling / stable


class AlertState(BaseModel):
    mode: str = "auto"
    active: bool = False
    safety_state: SafetyState = SafetyState.NORMAL
    led_color: LedColor = LedColor.GREEN
    buzzer: BuzzerMode = BuzzerMode.OFF
    muted_until: float | None = None
    reason: str = ""
    last_command_at: float | None = None
    last_ack_at: float | None = None
    pending_commands: int = 0
    simulated: bool = True


class SystemHealth(BaseModel):
    health_state: HealthState = HealthState.HEALTHY
    safety_state: SafetyState = SafetyState.NORMAL
    simulation_mode: bool = True
    qwen_configured: bool = False
    qwen_available: bool = False
    qwen_enabled: bool = True
    database_ok: bool = True
    serial_connected: bool = False
    serial_port: str = ""
    ws_clients: int = 0
    vision_backend: str = "simulated"
    uptime_seconds: float = 0.0
    degraded_modules: list[str] = Field(default_factory=list)
    server_time: float = 0.0


class SystemStatus(BaseModel):
    health: SystemHealth
    alert: AlertState
    occupancy: OccupancyUpdate
    environment: dict[str, float | None] = Field(default_factory=dict)
    active_events: int = 0
    pending_events: int = 0
