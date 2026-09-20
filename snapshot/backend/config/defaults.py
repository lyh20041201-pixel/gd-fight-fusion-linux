"""业务默认参数（阈值、时段、规则）。

这些值只是"出厂默认"，运行时保存在 SQLite settings / rules 表中，
可通过系统设置页与规则接口修改，不写死在业务代码里。
"""

from __future__ import annotations

from typing import Any

DEFAULT_THRESHOLDS: dict[str, Any] = {
    # CO2（ppm）：进入阈值 / 解除阈值（滞回）/ 持续时间（秒）
    "co2_warning": 1000.0,
    "co2_warning_clear": 850.0,
    "co2_critical": 1800.0,
    "co2_critical_clear": 1500.0,
    "co2_duration_seconds": 60.0,
    # 温湿度（仅提示）
    "temperature_high": 32.0,
    "temperature_low": 8.0,
    "humidity_high": 80.0,
    "humidity_low": 20.0,
    # 噪声（dB）
    "noise_warning": 75.0,
    "noise_clear": 65.0,
    "noise_duration_seconds": 120.0,
    # 光照（lux，用于课后判断辅助）
    "light_dark": 50.0,
    # 人数
    "occupancy_limit": 60,
    "occupancy_spike_delta": 8,
    "occupancy_spike_window_seconds": 10.0,
    "occupancy_median_window_seconds": 3.0,
    # 有人 / 无人判定
    "presence_frames_m": 4,  # N 帧中 M 帧成立
    "presence_frames_n": 6,
    "absence_seconds": 12.0,
    # 多摄像头人数矛盾
    "camera_conflict_delta": 3,
    "camera_conflict_duration_seconds": 20.0,
    # 摄像头遮挡（画面方差阈值）
    "obstruction_variance": 40.0,
    # 方差持续低于阈值多久才认定为遮挡（画面级判定）
    "obstruction_hold_seconds": 3.0,
    # 遮挡状态持续多久才建档为事件（规则级判定）
    "obstruction_duration_seconds": 15.0,
    # 设备心跳
    "heartbeat_timeout_seconds": 15.0,
    # 事件冷却（秒）
    "event_cooldown_seconds": 120.0,
}

DEFAULT_SCHEDULE: dict[str, Any] = {
    # 课后时间段：该时间之后仍检测到人员 -> after_hours_presence
    "class_start": "07:30",
    "class_end": "21:00",
    # 禁止时段（跨零点写法允许 start > end）
    "restricted_periods": [{"start": "23:00", "end": "05:30"}],
    # 课后持续多久才判定
    "after_hours_duration_seconds": 60.0,
}

# 规则定义。params 中的键与 DEFAULT_THRESHOLDS 对应，可在规则页单独覆盖。
DEFAULT_RULES: list[dict[str, Any]] = [
    {
        "rule_id": "smoke_detected",
        "name": "烟感触发",
        "description": "烟感继电器闭合后立即本地报警，不等待 Qwen，Qwen 也不能取消该报警。",
        "event_type": "smoke_detected",
        "risk_level": "critical",
        "params": {"cooldown_seconds": 60.0, "hard_alarm": True},
        "editable_params": ["cooldown_seconds"],
    },
    {
        "rule_id": "co2_critical",
        "name": "CO₂ 严重超标",
        "description": "CO₂ 超过严重阈值并持续一段时间后报警，低于解除阈值后自动解除。",
        "event_type": "co2_critical",
        "risk_level": "high",
        "params": {
            "enter": 1800.0,
            "clear": 1500.0,
            "duration_seconds": 60.0,
            "cooldown_seconds": 300.0,
        },
        "editable_params": ["enter", "clear", "duration_seconds", "cooldown_seconds"],
    },
    {
        "rule_id": "co2_warning",
        "name": "CO₂ 超标预警",
        "description": "CO₂ 超过预警阈值并持续一段时间后预警。",
        "event_type": "co2_warning",
        "risk_level": "medium",
        "params": {
            "enter": 1000.0,
            "clear": 850.0,
            "duration_seconds": 60.0,
            "cooldown_seconds": 300.0,
        },
        "editable_params": ["enter", "clear", "duration_seconds", "cooldown_seconds"],
    },
    {
        "rule_id": "noise_abnormal",
        "name": "噪声持续异常",
        "description": "噪声超过阈值并持续一段时间后预警，采用滑动窗口中位数抑制瞬时尖峰。",
        "event_type": "noise_abnormal",
        "risk_level": "medium",
        "params": {
            "enter": 75.0,
            "clear": 65.0,
            "duration_seconds": 120.0,
            "cooldown_seconds": 300.0,
        },
        "editable_params": ["enter", "clear", "duration_seconds", "cooldown_seconds"],
    },
    {
        "rule_id": "after_hours_presence",
        "name": "课后仍有人员",
        "description": "课后时间段内持续检测到人员。",
        "event_type": "after_hours_presence",
        "risk_level": "medium",
        "params": {"duration_seconds": 60.0, "cooldown_seconds": 600.0},
        "editable_params": ["duration_seconds", "cooldown_seconds"],
    },
    {
        "rule_id": "restricted_time_presence",
        "name": "禁止时段出现人员",
        "description": "禁止时段内检测到人员立即预警。",
        "event_type": "restricted_time_presence",
        "risk_level": "high",
        "params": {"duration_seconds": 15.0, "cooldown_seconds": 600.0},
        "editable_params": ["duration_seconds", "cooldown_seconds"],
    },
    {
        "rule_id": "over_capacity",
        "name": "教室超员",
        "description": "全局人数（最近若干秒中位数）超过上限并持续一段时间。",
        "event_type": "over_capacity",
        "risk_level": "medium",
        "params": {"limit": 60, "duration_seconds": 30.0, "cooldown_seconds": 300.0},
        "editable_params": ["limit", "duration_seconds", "cooldown_seconds"],
    },
    {
        "rule_id": "occupancy_spike",
        "name": "人数突然大幅变化",
        "description": "短窗口内人数变化超过阈值。",
        "event_type": "occupancy_spike",
        "risk_level": "low",
        "params": {
            "delta": 8,
            "window_seconds": 10.0,
            "cooldown_seconds": 180.0,
            "auto_clear_seconds": 60.0,
        },
        "editable_params": ["delta", "window_seconds", "cooldown_seconds"],
    },
    {
        "rule_id": "camera_count_conflict",
        "name": "多摄像头人数矛盾",
        "description": "重叠区域主副摄像头人数差异持续超过阈值，触发 Qwen 复核。",
        "event_type": "camera_count_conflict",
        "risk_level": "low",
        "params": {
            "delta": 3,
            "duration_seconds": 20.0,
            "cooldown_seconds": 300.0,
            "auto_clear_seconds": 30.0,
        },
        "editable_params": ["delta", "duration_seconds", "cooldown_seconds"],
    },
    {
        "rule_id": "camera_obstructed",
        "name": "摄像头被遮挡",
        "description": "画面方差长期过低，判定镜头被遮挡或失焦。",
        "event_type": "camera_obstructed",
        "risk_level": "medium",
        "params": {
            "variance": 40.0,
            "duration_seconds": 15.0,
            "cooldown_seconds": 300.0,
            "auto_clear_seconds": 20.0,
        },
        "editable_params": ["variance", "duration_seconds", "cooldown_seconds"],
    },
    {
        "rule_id": "device_offline",
        "name": "关键设备离线",
        "description": "网关、节点或摄像头心跳超时，进入系统故障（不影响安全状态判定）。",
        "event_type": "device_offline",
        "risk_level": "medium",
        "params": {
            "timeout_seconds": 15.0,
            "cooldown_seconds": 300.0,
            "auto_clear_seconds": 15.0,
        },
        "editable_params": ["timeout_seconds", "cooldown_seconds"],
    },
]

# 接口可修改的规则参数取值范围：键 -> (最小值, 最大值)。
# 未列出的键一律拒绝：非法值（负数持续时间、字符串）会让控制循环反复抛异常，
# 或让规则永远无法触发 / 永远无法解除。见 backend/schemas/api.py: RulePatch。
RULE_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "cooldown_seconds": (0.0, 86400.0),
    "duration_seconds": (0.0, 86400.0),
    "window_seconds": (1.0, 86400.0),
    "timeout_seconds": (1.0, 86400.0),
    "enter": (0.0, 100000.0),
    "clear": (0.0, 100000.0),
    "limit": (1.0, 10000.0),
    "delta": (1.0, 10000.0),
    "variance": (0.0, 100000.0),
}

# 设置页阈值的取值范围，含义同上。温度允许负值（未供暖教室）。
THRESHOLD_BOUNDS: dict[str, tuple[float, float]] = {
    "co2_warning": (0.0, 50000.0),
    "co2_warning_clear": (0.0, 50000.0),
    "co2_critical": (0.0, 50000.0),
    "co2_critical_clear": (0.0, 50000.0),
    "co2_duration_seconds": (0.0, 86400.0),
    "temperature_high": (-50.0, 100.0),
    "temperature_low": (-50.0, 100.0),
    "humidity_high": (0.0, 100.0),
    "humidity_low": (0.0, 100.0),
    "noise_warning": (0.0, 200.0),
    "noise_clear": (0.0, 200.0),
    "noise_duration_seconds": (0.0, 86400.0),
    "light_dark": (0.0, 200000.0),
    "occupancy_limit": (1.0, 10000.0),
    "occupancy_spike_delta": (1.0, 10000.0),
    "occupancy_spike_window_seconds": (1.0, 86400.0),
    "occupancy_median_window_seconds": (0.1, 86400.0),
    "presence_frames_m": (1.0, 100.0),
    "presence_frames_n": (1.0, 100.0),
    "absence_seconds": (0.0, 86400.0),
    "camera_conflict_delta": (1.0, 10000.0),
    "camera_conflict_duration_seconds": (0.0, 86400.0),
    "obstruction_variance": (0.0, 100000.0),
    "obstruction_hold_seconds": (0.0, 3600.0),
    "obstruction_duration_seconds": (0.0, 86400.0),
    "heartbeat_timeout_seconds": (1.0, 86400.0),
    "event_cooldown_seconds": (0.0, 86400.0),
}

# 作息表允许的键，以及每项时段最多允许的条数。
SCHEDULE_KEYS = frozenset(
    {"class_start", "class_end", "restricted_periods", "after_hours_duration_seconds"}
)
MAX_RESTRICTED_PERIODS = 10

# 需要触发 Qwen 复核的事件类型（Qwen 只做解释与复核，不作为唯一判断来源）
QWEN_REVIEW_EVENT_TYPES = {
    "after_hours_presence",
    "restricted_time_presence",
    "camera_count_conflict",
    "occupancy_spike",
    "camera_obstructed",
    "co2_critical",
    "over_capacity",
}

# 默认摄像头区域（模拟模式下自动创建）
# (区域名, 是否为该区域主摄像头)。区域相同的多路摄像头视为重叠覆盖，
# 只有主摄像头参与全局人数求和，副摄像头用于交叉校验（人数矛盾检测）。
DEFAULT_CAMERA_REGIONS = [
    ("讲台与前排", True),
    ("中部座位区", True),
    ("后排与门口", True),
    ("后排与门口", False),
]
