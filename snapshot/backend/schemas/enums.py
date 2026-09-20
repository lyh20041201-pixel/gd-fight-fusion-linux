"""系统内统一使用的枚举。

注意：教室安全状态（SafetyState）与系统健康状态（HealthState）是两套独立维度。
"CO2 超标" 属于安全状态；"CO2 传感器离线" 属于系统健康状态。
"""

from __future__ import annotations

from enum import Enum


class SafetyState(str, Enum):
    """教室安全状态。"""

    NORMAL = "NORMAL"
    NOTICE = "NOTICE"
    WARNING = "WARNING"
    ALARM = "ALARM"

    @property
    def severity(self) -> int:
        return {"NORMAL": 0, "NOTICE": 1, "WARNING": 2, "ALARM": 3}[self.value]


class HealthState(str, Enum):
    """系统健康状态。"""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAULT = "FAULT"

    @property
    def severity(self) -> int:
        return {"HEALTHY": 0, "DEGRADED": 1, "FAULT": 2}[self.value]


class ModuleState(str, Enum):
    """单个模块状态，对应界面颜色。"""

    INITIALIZING = "INITIALIZING"  # 蓝色
    ONLINE = "ONLINE"  # 绿色
    WARNING = "WARNING"  # 黄色
    ALARM = "ALARM"  # 红色
    OFFLINE = "OFFLINE"  # 灰色
    FAULT = "FAULT"  # 紫色/故障色


class ModuleType(str, Enum):
    GATEWAY = "gateway"
    NODE = "node"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    DISPLAY = "display"
    CAMERA = "camera"
    SERVICE = "service"


class SensorKind(str, Enum):
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    CO2 = "co2"
    LIGHT = "light"
    NOISE = "noise"
    SMOKE = "smoke"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def severity(self) -> int:
        return {"low": 0, "medium": 1, "high": 2, "critical": 3}[self.value]


class EventStatus(str, Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    FALSE_ALARM = "FALSE_ALARM"
    RESOLVED = "RESOLVED"


class EventSource(str, Enum):
    VISION = "vision"
    SENSOR = "sensor"
    FUSION = "fusion"
    SYSTEM = "system"
    MANUAL = "manual"


class EventType(str, Enum):
    AFTER_HOURS_PRESENCE = "after_hours_presence"
    RESTRICTED_TIME_PRESENCE = "restricted_time_presence"
    OVER_CAPACITY = "over_capacity"
    OCCUPANCY_SPIKE = "occupancy_spike"
    CAMERA_COUNT_CONFLICT = "camera_count_conflict"
    SMOKE_DETECTED = "smoke_detected"
    CO2_CRITICAL = "co2_critical"
    CO2_WARNING = "co2_warning"
    NOISE_ABNORMAL = "noise_abnormal"
    CAMERA_OBSTRUCTED = "camera_obstructed"
    DEVICE_OFFLINE = "device_offline"
    # 预留（第一版不触发，规则引擎已留出接入点）
    PERSON_FALL = "person_fall"
    SUSPECTED_FIGHT = "suspected_fight"
    LONG_STATIC = "long_static"
    FAST_RUNNING = "fast_running"
    ABNORMAL_GATHERING = "abnormal_gathering"
    RESTRICTED_AREA_INTRUSION = "restricted_area_intrusion"


EVENT_TYPE_LABELS: dict[str, str] = {
    EventType.AFTER_HOURS_PRESENCE.value: "课后仍有人员",
    EventType.RESTRICTED_TIME_PRESENCE.value: "禁止时段出现人员",
    EventType.OVER_CAPACITY.value: "教室超员",
    EventType.OCCUPANCY_SPIKE.value: "人数突然大幅变化",
    EventType.CAMERA_COUNT_CONFLICT.value: "多摄像头人数矛盾",
    EventType.SMOKE_DETECTED.value: "烟感触发",
    EventType.CO2_CRITICAL.value: "CO₂ 严重超标",
    EventType.CO2_WARNING.value: "CO₂ 超标预警",
    EventType.NOISE_ABNORMAL.value: "噪声持续异常",
    EventType.CAMERA_OBSTRUCTED.value: "摄像头被遮挡",
    EventType.DEVICE_OFFLINE.value: "关键设备离线",
    EventType.PERSON_FALL.value: "人员倒地",
    EventType.SUSPECTED_FIGHT.value: "疑似打架",
    EventType.LONG_STATIC.value: "长时间静止",
    EventType.FAST_RUNNING.value: "快速奔跑",
    EventType.ABNORMAL_GATHERING.value: "异常聚集",
    EventType.RESTRICTED_AREA_INTRUSION.value: "危险区域闯入",
}


class AlertMode(str, Enum):
    AUTO = "auto"
    MANUAL_TEST = "manual_test"


class BuzzerMode(str, Enum):
    OFF = "off"
    CONTINUOUS = "continuous"
    PULSE = "pulse"


class LedColor(str, Enum):
    OFF = "off"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    RED = "red"


class WsEventType(str, Enum):
    SNAPSHOT = "snapshot"
    SENSOR_UPDATE = "sensor_update"
    MODULE_STATUS = "module_status"
    CAMERA_STATUS = "camera_status"
    OCCUPANCY_UPDATE = "occupancy_update"
    RISK_EVENT_CREATED = "risk_event_created"
    RISK_EVENT_UPDATED = "risk_event_updated"
    ALERT_STATE = "alert_state"
    SYSTEM_HEALTH = "system_health"
    LOG = "log"
