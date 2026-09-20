"""规则条件原语。

提供：数值阈值 + 滞回、持续时间、N 帧中 M 帧、滑动窗口中位数、冷却与去重。
所有参数都来自可配置的规则参数，不写死在业务逻辑里。
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


class MedianWindow:
    """滑动窗口中位数，用于抑制瞬时抖动。"""

    def __init__(self, window_seconds: float = 3.0, maxlen: int = 600) -> None:
        self.window_seconds = window_seconds
        self._values: deque[tuple[float, float]] = deque(maxlen=maxlen)

    def push(self, value: float, timestamp: float | None = None) -> None:
        self._values.append((timestamp or time.time(), float(value)))

    def median(self, default: float = 0.0) -> float:
        cutoff = time.time() - self.window_seconds
        values = [v for ts, v in self._values if ts >= cutoff]
        if not values:
            return float(self._values[-1][1]) if self._values else default
        return float(statistics.median(values))

    def latest(self, default: float | None = None) -> float | None:
        return self._values[-1][1] if self._values else default

    def span(self) -> float:
        cutoff = time.time() - self.window_seconds
        values = [v for ts, v in self._values if ts >= cutoff]
        if len(values) < 2:
            return 0.0
        return max(values) - min(values)

    def clear(self) -> None:
        self._values.clear()


class NofM:
    """最近 N 次判定中至少 M 次成立。"""

    def __init__(self, n: int = 6, m: int = 4) -> None:
        self.n = max(1, n)
        self.m = max(1, min(m, self.n))
        self._flags: deque[bool] = deque(maxlen=self.n)

    def push(self, flag: bool) -> bool:
        self._flags.append(bool(flag))
        return self.satisfied

    @property
    def satisfied(self) -> bool:
        return sum(self._flags) >= self.m

    @property
    def all_false(self) -> bool:
        return len(self._flags) == self._flags.maxlen and not any(self._flags)

    def configure(self, n: int, m: int) -> None:
        if n != self.n:
            self.n = max(1, n)
            self._flags = deque(self._flags, maxlen=self.n)
        self.m = max(1, min(m, self.n))

    def clear(self) -> None:
        self._flags.clear()


@dataclass
class HysteresisLatch:
    """阈值 + 滞回 + 持续时间的状态机。

    condition_value >= enter 持续 duration 秒 -> 触发；
    condition_value <= clear -> 自动解除（clear 可低于 enter 形成滞回）。
    """

    enter: float
    clear: float
    duration: float = 0.0
    invert: bool = False  # True 表示"低于阈值"触发
    active: bool = False
    pending_since: float | None = None
    activated_at: float | None = None
    cleared_at: float | None = None

    def _over(self, value: float) -> bool:
        return value <= self.enter if self.invert else value >= self.enter

    def _under(self, value: float) -> bool:
        return value >= self.clear if self.invert else value <= self.clear

    def update(self, value: float, now: float | None = None) -> str:
        """返回 'trigger' / 'clear' / 'idle' / 'active' / 'pending'。"""
        now = now or time.time()
        if self._over(value):
            if self.active:
                return "active"
            if self.pending_since is None:
                self.pending_since = now
            if now - self.pending_since >= self.duration:
                self.active = True
                self.activated_at = now
                self.pending_since = None
                return "trigger"
            return "pending"
        if self._under(value):
            self.pending_since = None
            if self.active:
                self.active = False
                self.cleared_at = now
                return "clear"
            return "idle"
        # 介于 enter 与 clear 之间：保持当前状态（滞回区）
        self.pending_since = None
        return "active" if self.active else "idle"

    def configure(self, enter: float, clear: float, duration: float) -> None:
        self.enter, self.clear, self.duration = enter, clear, duration

    def reset(self) -> None:
        self.active = False
        self.pending_since = None


@dataclass
class Cooldown:
    """事件冷却与去重：冷却期内不重复创建同类事件。"""

    seconds: float = 120.0
    last_fired: float | None = None

    def ready(self, now: float | None = None) -> bool:
        if self.last_fired is None:
            return True
        return (now or time.time()) - self.last_fired >= self.seconds

    def fire(self, now: float | None = None) -> None:
        self.last_fired = now or time.time()

    def remaining(self, now: float | None = None) -> float:
        if self.last_fired is None:
            return 0.0
        return max(0.0, self.seconds - ((now or time.time()) - self.last_fired))


@dataclass
class RuleState:
    rule_id: str
    active: bool = False
    pending_since: float | None = None
    last_triggered_at: float | None = None
    event_id: str | None = None
    basis: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.active:
            return "active"
        return "pending" if self.pending_since else "idle"


def parse_hhmm(text: str, default: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    try:
        hh, mm = text.split(":")
        return int(hh), int(mm)
    except (ValueError, AttributeError):
        return default


def minutes_of(ts: float) -> int:
    lt = time.localtime(ts)
    return lt.tm_hour * 60 + lt.tm_min


def in_time_window(ts: float, start: str, end: str) -> bool:
    """支持跨零点的时间窗口判定。"""
    sh, sm = parse_hhmm(start, (0, 0))
    eh, em = parse_hhmm(end, (23, 59))
    start_min, end_min = sh * 60 + sm, eh * 60 + em
    current = minutes_of(ts)
    if start_min <= end_min:
        return start_min <= current <= end_min
    return current >= start_min or current <= end_min


def is_after_hours(ts: float, class_start: str, class_end: str) -> bool:
    """课后 = 不在上课时间段内。"""
    return not in_time_window(ts, class_start, class_end)
