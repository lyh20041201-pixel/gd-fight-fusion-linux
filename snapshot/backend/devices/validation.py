"""传感器数据有效性校验。

对应开题报告第三章"（2）数据预处理·文本侧"：
"对传感器读数进行有效性校验与单位归一，剔除因丢包或校验失败产生的无效值"。

设计约束（不可违反）：
- 无效值一律**丢弃**，绝不用 0 或"正常值"替换 —— 无数据就是无数据；
- NaN / ±Inf 必须拦在入口，否则会污染滑动窗口中位数，
  并让 json.dumps 产出 `NaN` 字面量（非法 JSON）打断前端 WebSocket 解析；
- 量程为"物理传感器可能输出的范围"，不是业务阈值。
  例如 CO₂ 报警阈值 1000 ppm 属于业务阈值，而 0~10000 ppm 才是量程。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# 各传感器的物理量程。依据本项目实际选型的器件手册：
#   温湿度 SHT3x/DHT 系列、CO₂ MH-Z19C、噪声 INMP441、烟感 JTY-GD-301AL（继电器 0/1）
DEFAULT_SENSOR_RANGES: dict[str, tuple[float, float]] = {
    "temperature": (-40.0, 85.0),      # °C
    "humidity": (0.0, 100.0),          # %RH
    "co2": (0.0, 10000.0),             # ppm
    "light": (0.0, 200000.0),          # lux（直射日光量级）
    "noise": (0.0, 140.0),             # dB
    "smoke": (0.0, 1.0),               # 继电器输出，仅 0 / 1
}

# 超过这个时间没有新读数，则认为该传感器数据已陈旧，不再参与判定
DEFAULT_STALE_SECONDS = 30.0


@dataclass(frozen=True)
class ValidationResult:
    """单条读数的校验结果。"""

    kind: str
    raw: Any
    value: float | None
    valid: bool
    reason: str | None = None

    @property
    def invalid(self) -> bool:
        return not self.valid


def _as_number(value: Any) -> float | None:
    """只接受真正的数值类型。

    布尔值单独放行并归一为 0/1：烟感等开关量在 JSON 中写成 true/false 是合法上报。
    字符串一律拒绝 —— 固件应当上报数值，字符串意味着协议出错，不能静默容忍。
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def validate_reading(
    kind: str,
    value: Any,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> ValidationResult:
    """校验单条传感器读数。"""
    number = _as_number(value)
    if number is None:
        return ValidationResult(
            kind, value, None, False, f"类型非法（{type(value).__name__}），期望数值"
        )
    if math.isnan(number):
        return ValidationResult(kind, value, None, False, "读数为 NaN")
    if math.isinf(number):
        return ValidationResult(kind, value, None, False, "读数为无穷大")

    table = ranges if ranges is not None else DEFAULT_SENSOR_RANGES
    bounds = table.get(kind)
    if bounds is None:
        # 未知种类：只保证是有限数，不做量程判断（允许固件扩展新传感器）
        return ValidationResult(kind, value, number, True, None)

    low, high = bounds
    if number < low or number > high:
        return ValidationResult(
            kind, value, None, False, f"超出量程 [{low:g}, {high:g}]：{number:g}"
        )
    return ValidationResult(kind, value, number, True, None)


def is_stale(updated_at: float | None, now: float, stale_seconds: float = DEFAULT_STALE_SECONDS) -> bool:
    """读数是否已陈旧。从未收到过读数（None）也算陈旧。"""
    if updated_at is None:
        return True
    return (now - updated_at) > stale_seconds


def finite_only(values: dict[str, float]) -> dict[str, float]:
    """兜底过滤：把任何非有限数从下发给上层/前端的字典中剔除。

    校验已经在入口拦了一道，这里是第二道防线 —— 一旦有新的写入路径绕过校验，
    也不至于让非法 JSON 流到 WebSocket。
    """
    return {
        key: value
        for key, value in values.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }
