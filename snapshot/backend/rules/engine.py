"""本地规则引擎。

特点：
- 阈值、持续时间、滞回、冷却全部来自可配置参数（rules 表 / settings 表）；
- 支持自动解除与人工确认两类结束方式；
- 烟感等硬报警立即生效，不等待 Qwen，Qwen 也无法取消；
- 引擎完全本地运行，Qwen 断网或异常都不影响判定。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..config.defaults import DEFAULT_RULES
from ..config.runtime import RuntimeConfig
from ..schemas.enums import EVENT_TYPE_LABELS, RiskLevel, SafetyState
from .conditions import (
    Cooldown,
    HysteresisLatch,
    MedianWindow,
    NofM,
    RuleState,
    in_time_window,
    is_after_hours,
)

logger = logging.getLogger(__name__)

RISK_TO_SAFETY = {
    RiskLevel.CRITICAL: SafetyState.ALARM,
    RiskLevel.HIGH: SafetyState.ALARM,
    RiskLevel.MEDIUM: SafetyState.WARNING,
    RiskLevel.LOW: SafetyState.NOTICE,
}

# 这些事件属于"系统健康"范畴，不参与教室安全状态计算
HEALTH_ONLY_EVENTS = {"device_offline", "camera_obstructed"}


@dataclass
class CameraSnapshot:
    camera_id: str
    online: bool
    obstructed: bool
    person_count: int
    region: str = ""
    track_labels: list[str] = field(default_factory=list)
    low_confidence: bool = False


@dataclass
class ClassroomSnapshot:
    timestamp: float
    sensors: dict[str, float] = field(default_factory=dict)
    sensor_online: dict[str, bool] = field(default_factory=dict)
    occupancy_total: int = 0
    occupancy_raw: int = 0
    occupancy_spike: int = 0
    cameras: list[CameraSnapshot] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    offline_devices: list[str] = field(default_factory=list)
    offline_cameras: list[str] = field(default_factory=list)


@dataclass
class RuleOutcome:
    kind: str  # trigger / clear
    rule_id: str
    event_type: str
    risk_level: RiskLevel
    title: str
    description: str
    basis: dict[str, Any] = field(default_factory=dict)
    camera_id: str | None = None
    track_ids: list[str] = field(default_factory=list)
    hard_alarm: bool = False
    needs_qwen: bool = False
    source: str = "fusion"


class RuleEngine:
    def __init__(self, runtime: RuntimeConfig, rules: list[dict[str, Any]] | None = None) -> None:
        self.runtime = runtime
        self._rules: dict[str, dict[str, Any]] = {}
        for rule in rules or DEFAULT_RULES:
            self._rules[rule["rule_id"]] = dict(rule)
        self._states: dict[str, RuleState] = {
            rule_id: RuleState(rule_id=rule_id) for rule_id in self._rules
        }
        # 冷却按「规则:目标」独立计时。摄像头遮挡、区域矛盾、设备离线这类规则
        # 会同时作用于多个目标，共用一个冷却会导致第一个目标触发后其余目标被
        # 一起冷却掉，从而漏报（例如 CAM-01 遮挡后 CAM-02 遮挡不再建档）。
        self._cooldowns: dict[str, Cooldown] = {
            rule_id: Cooldown(float(rule.get("params", {}).get("cooldown_seconds", 120.0)))
            for rule_id, rule in self._rules.items()
        }
        self._latches: dict[str, HysteresisLatch] = {}
        self._co2_window = MedianWindow(10.0)
        self._noise_window = MedianWindow(15.0)
        self._occupancy_window = MedianWindow(
            float(runtime.threshold("occupancy_median_window_seconds", 3.0))
        )
        self._presence = NofM(
            int(runtime.threshold("presence_frames_n", 6)),
            int(runtime.threshold("presence_frames_m", 4)),
        )
        self._presence_active = False
        self._absent_since: float | None = None
        self._present_since: float | None = None
        self._camera_obstruction_since: dict[str, float] = {}
        self._conflict_since: dict[str, float] = {}
        self._offline_since: dict[str, float] = {}
        self._last_safety = SafetyState.NORMAL
        self._active_reasons: dict[str, str] = {}

    # ================= 配置 =================
    def rules(self) -> list[dict[str, Any]]:
        out = []
        for rule_id, rule in self._rules.items():
            state = self._states[rule_id]
            item = dict(rule)
            item["state"] = state.label
            item["last_triggered_at"] = state.last_triggered_at
            out.append(item)
        return out

    def rule(self, rule_id: str) -> dict[str, Any] | None:
        return self._rules.get(rule_id)

    def update_rule(
        self,
        rule_id: str,
        *,
        enabled: bool | None = None,
        risk_level: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        rule = self._rules.get(rule_id)
        if rule is None:
            return None
        if enabled is not None:
            rule["enabled"] = bool(enabled)
        if risk_level:
            rule["risk_level"] = risk_level
        if params:
            # 阈值同步时可能带入 None（对应阈值项缺失），float(None) 会炸掉整次更新
            params = {k: v for k, v in params.items() if v is not None}
        if params:
            merged = dict(rule.get("params", {}))
            merged.update(params)
            rule["params"] = merged
            if "cooldown_seconds" in params:
                seconds = float(params["cooldown_seconds"])
                # 同步该规则下的全部按目标冷却器，而不只是无目标的那一个
                for key, cooldown in self._cooldowns.items():
                    if key == rule_id or key.startswith(f"{rule_id}:"):
                        cooldown.seconds = seconds
            latch = self._latches.get(rule_id)
            if latch is not None:
                latch.configure(
                    float(merged.get("enter", latch.enter)),
                    float(merged.get("clear", latch.clear)),
                    float(merged.get("duration_seconds", latch.duration)),
                )
        return rule

    def _param(self, rule_id: str, key: str, default: Any) -> Any:
        return self._rules.get(rule_id, {}).get("params", {}).get(key, default)

    def _enabled(self, rule_id: str) -> bool:
        return bool(self._rules.get(rule_id, {}).get("enabled", True))

    def _risk(self, rule_id: str, fallback: str = "medium") -> RiskLevel:
        return RiskLevel(self._rules.get(rule_id, {}).get("risk_level", fallback))

    def _cooldown(self, rule_id: str, target: str | None = None) -> Cooldown:
        """取该规则（可按目标细分）的冷却器，不存在时按规则参数惰性创建。"""
        key = rule_id if not target else f"{rule_id}:{target}"
        cooldown = self._cooldowns.get(key)
        if cooldown is None:
            cooldown = Cooldown(float(self._param(rule_id, "cooldown_seconds", 120.0)))
            self._cooldowns[key] = cooldown
        return cooldown

    def _latch(self, rule_id: str, enter: float, clear: float, duration: float,
               invert: bool = False) -> HysteresisLatch:
        latch = self._latches.get(rule_id)
        if latch is None:
            latch = HysteresisLatch(enter=enter, clear=clear, duration=duration, invert=invert)
            self._latches[rule_id] = latch
        else:
            latch.configure(enter, clear, duration)
        return latch

    # ================= 评估 =================
    def evaluate(self, snapshot: ClassroomSnapshot) -> list[RuleOutcome]:
        outcomes: list[RuleOutcome] = []
        now = snapshot.timestamp

        self._update_presence(snapshot, now)

        outcomes += self._rule_smoke(snapshot, now)
        outcomes += self._rule_co2(snapshot, now)
        outcomes += self._rule_noise(snapshot, now)
        outcomes += self._rule_presence_schedule(snapshot, now)
        outcomes += self._rule_capacity(snapshot, now)
        outcomes += self._rule_spike(snapshot, now)
        outcomes += self._rule_conflict(snapshot, now)
        outcomes += self._rule_obstruction(snapshot, now)
        outcomes += self._rule_device_offline(snapshot, now)
        outcomes += self._expire_transient(snapshot, now)

        for outcome in outcomes:
            state = self._states.setdefault(outcome.rule_id, RuleState(outcome.rule_id))
            if outcome.kind == "trigger":
                state.active = True
                state.last_triggered_at = now
                state.basis = outcome.basis
                self._active_reasons[outcome.rule_id] = outcome.title
            elif outcome.kind == "clear":
                state.active = False
                state.basis = {}
                self._active_reasons.pop(outcome.rule_id, None)
        return outcomes

    # ---------- 瞬时类规则的自动解除 ----------
    def _expire_transient(self, snapshot: ClassroomSnapshot, now: float) -> list[RuleOutcome]:
        """瞬时事件（突变 / 遮挡 / 矛盾 / 离线）在条件消失后自动解除。

        自动解除只影响"当前状态"，已经建档的事件仍需人工确认或标记已处理。
        """
        outcomes: list[RuleOutcome] = []
        conditions = {
            "occupancy_spike": lambda: snapshot.occupancy_spike
            >= int(self._param("occupancy_spike", "delta", 8)),
            "camera_obstructed": lambda: any(c.obstructed for c in snapshot.cameras),
            "camera_count_conflict": lambda: bool(snapshot.conflicts),
            "device_offline": lambda: bool(
                snapshot.offline_devices or snapshot.offline_cameras
            ),
        }
        for rule_id, still_true in conditions.items():
            state = self._states.get(rule_id)
            if state is None or not state.active:
                continue
            hold = float(self._param(rule_id, "auto_clear_seconds", 30.0))
            if still_true():
                continue
            if state.last_triggered_at and now - state.last_triggered_at < hold:
                continue
            rule = self._rules.get(rule_id, {})
            outcomes.append(
                RuleOutcome(
                    kind="clear",
                    rule_id=rule_id,
                    event_type=rule.get("event_type", rule_id),
                    risk_level=self._risk(rule_id),
                    title=EVENT_TYPE_LABELS.get(rule.get("event_type", rule_id), rule_id),
                    description="触发条件已消失，状态自动解除。",
                    basis={"auto_clear_seconds": hold},
                    source="system",
                )
            )
        return outcomes

    # ---------- 人员在场判定 ----------
    def _update_presence(self, snapshot: ClassroomSnapshot, now: float) -> None:
        self._presence.configure(
            int(self.runtime.threshold("presence_frames_n", 6)),
            int(self.runtime.threshold("presence_frames_m", 4)),
        )
        self._occupancy_window.window_seconds = float(
            self.runtime.threshold("occupancy_median_window_seconds", 3.0)
        )
        self._occupancy_window.push(snapshot.occupancy_raw, now)
        has_person = snapshot.occupancy_raw > 0
        self._presence.push(has_person)

        absence_seconds = float(self.runtime.threshold("absence_seconds", 12.0))
        if self._presence.satisfied:
            self._absent_since = None
            if not self._presence_active:
                self._presence_active = True
                self._present_since = now
        elif not has_person:
            self._absent_since = self._absent_since or now
            if self._presence_active and now - self._absent_since >= absence_seconds:
                self._presence_active = False
                self._present_since = None

    @property
    def presence(self) -> bool:
        return self._presence_active

    # ---------- 烟感（硬报警） ----------
    def _rule_smoke(self, snapshot: ClassroomSnapshot, now: float) -> list[RuleOutcome]:
        rule_id = "smoke_detected"
        if not self._enabled(rule_id):
            return []
        state = self._states[rule_id]
        value = snapshot.sensors.get("smoke")
        if value is None:
            return []
        triggered = value >= 1
        if triggered and not state.active:
            if not self._cooldown(rule_id).ready(now):
                return []
            self._cooldown(rule_id).fire(now)
            return [
                RuleOutcome(
                    kind="trigger",
                    rule_id=rule_id,
                    event_type="smoke_detected",
                    risk_level=self._risk(rule_id, "critical"),
                    title=EVENT_TYPE_LABELS["smoke_detected"],
                    description="烟感继电器输出触发，已立即启动本地声光报警（不依赖 Qwen）。",
                    basis={"sensor": "smoke", "value": value, "hard_alarm": True},
                    hard_alarm=True,
                    needs_qwen=False,
                    source="sensor",
                )
            ]
        if not triggered and state.active:
            return [
                RuleOutcome(
                    kind="clear",
                    rule_id=rule_id,
                    event_type="smoke_detected",
                    risk_level=self._risk(rule_id, "critical"),
                    title=EVENT_TYPE_LABELS["smoke_detected"],
                    description="烟感恢复，硬报警条件解除（事件仍需人工确认）。",
                    basis={"sensor": "smoke", "value": value},
                    source="sensor",
                )
            ]
        return []

    # ---------- CO2 ----------
    def _rule_co2(self, snapshot: ClassroomSnapshot, now: float) -> list[RuleOutcome]:
        value = snapshot.sensors.get("co2")
        if value is None:
            return []
        self._co2_window.push(value, now)
        median = self._co2_window.median(value)
        outcomes: list[RuleOutcome] = []
        for rule_id, default_risk in (("co2_critical", "high"), ("co2_warning", "medium")):
            if not self._enabled(rule_id):
                continue
            latch = self._latch(
                rule_id,
                float(self._param(rule_id, "enter", 1000.0)),
                float(self._param(rule_id, "clear", 850.0)),
                float(self._param(rule_id, "duration_seconds", 60.0)),
            )
            result = latch.update(median, now)
            if result == "trigger" and self._cooldown(rule_id).ready(now):
                self._cooldown(rule_id).fire(now)
                outcomes.append(
                    RuleOutcome(
                        kind="trigger",
                        rule_id=rule_id,
                        event_type=rule_id,
                        risk_level=self._risk(rule_id, default_risk),
                        title=EVENT_TYPE_LABELS[rule_id],
                        description=(
                            f"CO₂ 中位数 {median:.0f} ppm 超过阈值 {latch.enter:.0f} ppm "
                            f"并持续 {latch.duration:.0f} 秒。"
                        ),
                        basis={
                            "sensor": "co2",
                            "median": round(median, 1),
                            "enter": latch.enter,
                            "clear": latch.clear,
                            "duration_seconds": latch.duration,
                        },
                        needs_qwen=rule_id == "co2_critical",
                        source="sensor",
                    )
                )
            elif result == "clear":
                outcomes.append(
                    RuleOutcome(
                        kind="clear",
                        rule_id=rule_id,
                        event_type=rule_id,
                        risk_level=self._risk(rule_id, default_risk),
                        title=EVENT_TYPE_LABELS[rule_id],
                        description=f"CO₂ 已降至解除阈值 {latch.clear:.0f} ppm 以下。",
                        basis={"sensor": "co2", "median": round(median, 1)},
                        source="sensor",
                    )
                )
        return outcomes

    # ---------- 噪声 ----------
    def _rule_noise(self, snapshot: ClassroomSnapshot, now: float) -> list[RuleOutcome]:
        rule_id = "noise_abnormal"
        value = snapshot.sensors.get("noise")
        if value is None or not self._enabled(rule_id):
            return []
        self._noise_window.push(value, now)
        median = self._noise_window.median(value)
        latch = self._latch(
            rule_id,
            float(self._param(rule_id, "enter", 75.0)),
            float(self._param(rule_id, "clear", 65.0)),
            float(self._param(rule_id, "duration_seconds", 120.0)),
        )
        result = latch.update(median, now)
        if result == "trigger" and self._cooldown(rule_id).ready(now):
            self._cooldown(rule_id).fire(now)
            return [
                RuleOutcome(
                    kind="trigger",
                    rule_id=rule_id,
                    event_type="noise_abnormal",
                    risk_level=self._risk(rule_id, "medium"),
                    title=EVENT_TYPE_LABELS["noise_abnormal"],
                    description=(
                        f"噪声中位数 {median:.1f} dB 超过阈值 {latch.enter:.0f} dB "
                        f"并持续 {latch.duration:.0f} 秒。"
                    ),
                    basis={"sensor": "noise", "median": round(median, 1), "enter": latch.enter},
                    source="sensor",
                )
            ]
        if result == "clear":
            return [
                RuleOutcome(
                    kind="clear",
                    rule_id=rule_id,
                    event_type="noise_abnormal",
                    risk_level=self._risk(rule_id, "medium"),
                    title=EVENT_TYPE_LABELS["noise_abnormal"],
                    description="噪声已恢复正常水平。",
                    basis={"sensor": "noise", "median": round(median, 1)},
                    source="sensor",
                )
            ]
        return []

    # ---------- 课后 / 禁止时段有人 ----------
    def _rule_presence_schedule(
        self, snapshot: ClassroomSnapshot, now: float
    ) -> list[RuleOutcome]:
        outcomes: list[RuleOutcome] = []
        schedule = self.runtime.schedule
        track_ids = [
            label
            for cam in snapshot.cameras
            for label in (f"{cam.camera_id}-{t}" for t in cam.track_labels)
        ]
        primary_camera = next(
            (c.camera_id for c in snapshot.cameras if c.online and c.person_count > 0), None
        )

        # 课后仍有人员
        rule_id = "after_hours_presence"
        if self._enabled(rule_id):
            state = self._states[rule_id]
            after_hours = is_after_hours(
                now, schedule.get("class_start", "07:30"), schedule.get("class_end", "21:00")
            )
            duration = float(self._param(rule_id, "duration_seconds", 60.0))
            condition = after_hours and self._presence_active
            if condition:
                state.pending_since = state.pending_since or now
                if (
                    not state.active
                    and now - state.pending_since >= duration
                    and self._cooldown(rule_id).ready(now)
                ):
                    self._cooldown(rule_id).fire(now)
                    outcomes.append(
                        RuleOutcome(
                            kind="trigger",
                            rule_id=rule_id,
                            event_type=rule_id,
                            risk_level=self._risk(rule_id, "medium"),
                            title=EVENT_TYPE_LABELS[rule_id],
                            description=(
                                f"课后时间仍检测到 {snapshot.occupancy_total} 人，"
                                f"持续超过 {duration:.0f} 秒。"
                            ),
                            basis={
                                "occupancy": snapshot.occupancy_total,
                                "class_end": schedule.get("class_end"),
                                "duration_seconds": duration,
                                "presence_rule": "N 帧中 M 帧",
                            },
                            camera_id=primary_camera,
                            track_ids=track_ids[:12],
                            needs_qwen=True,
                            source="vision",
                        )
                    )
            else:
                state.pending_since = None
                if state.active:
                    outcomes.append(
                        RuleOutcome(
                            kind="clear",
                            rule_id=rule_id,
                            event_type=rule_id,
                            risk_level=self._risk(rule_id, "medium"),
                            title=EVENT_TYPE_LABELS[rule_id],
                            description="课后人员已离开或已进入上课时段。",
                            basis={"occupancy": snapshot.occupancy_total},
                            source="vision",
                        )
                    )

        # 禁止时段出现人员
        rule_id = "restricted_time_presence"
        if self._enabled(rule_id):
            state = self._states[rule_id]
            periods = schedule.get("restricted_periods", [])
            in_restricted = any(
                in_time_window(now, p.get("start", "23:00"), p.get("end", "05:30"))
                for p in periods
            )
            duration = float(self._param(rule_id, "duration_seconds", 15.0))
            condition = in_restricted and self._presence_active
            if condition:
                state.pending_since = state.pending_since or now
                if (
                    not state.active
                    and now - state.pending_since >= duration
                    and self._cooldown(rule_id).ready(now)
                ):
                    self._cooldown(rule_id).fire(now)
                    outcomes.append(
                        RuleOutcome(
                            kind="trigger",
                            rule_id=rule_id,
                            event_type=rule_id,
                            risk_level=self._risk(rule_id, "high"),
                            title=EVENT_TYPE_LABELS[rule_id],
                            description=f"禁止时段检测到 {snapshot.occupancy_total} 人。",
                            basis={
                                "occupancy": snapshot.occupancy_total,
                                "restricted_periods": periods,
                            },
                            camera_id=primary_camera,
                            track_ids=track_ids[:12],
                            needs_qwen=True,
                            source="vision",
                        )
                    )
            else:
                state.pending_since = None
                if state.active:
                    outcomes.append(
                        RuleOutcome(
                            kind="clear",
                            rule_id=rule_id,
                            event_type=rule_id,
                            risk_level=self._risk(rule_id, "high"),
                            title=EVENT_TYPE_LABELS[rule_id],
                            description="禁止时段人员已离开。",
                            basis={},
                            source="vision",
                        )
                    )
        return outcomes

    # ---------- 超员 ----------
    def _rule_capacity(self, snapshot: ClassroomSnapshot, now: float) -> list[RuleOutcome]:
        rule_id = "over_capacity"
        if not self._enabled(rule_id):
            return []
        limit = float(self._param(rule_id, "limit", self.runtime.threshold("occupancy_limit", 60)))
        duration = float(self._param(rule_id, "duration_seconds", 30.0))
        latch = self._latch(rule_id, limit, max(0.0, limit - 3), duration)
        median = self._occupancy_window.median(snapshot.occupancy_total)
        result = latch.update(median, now)
        if result == "trigger" and self._cooldown(rule_id).ready(now):
            self._cooldown(rule_id).fire(now)
            return [
                RuleOutcome(
                    kind="trigger",
                    rule_id=rule_id,
                    event_type=rule_id,
                    risk_level=self._risk(rule_id, "medium"),
                    title=EVENT_TYPE_LABELS[rule_id],
                    description=f"教室人数中位数 {median:.0f} 超过上限 {limit:.0f}。",
                    basis={"median": median, "limit": limit, "duration_seconds": duration},
                    camera_id=None,
                    needs_qwen=True,
                    source="fusion",
                )
            ]
        if result == "clear":
            return [
                RuleOutcome(
                    kind="clear",
                    rule_id=rule_id,
                    event_type=rule_id,
                    risk_level=self._risk(rule_id, "medium"),
                    title=EVENT_TYPE_LABELS[rule_id],
                    description="人数已回落至上限以下。",
                    basis={"median": median, "limit": limit},
                    source="fusion",
                )
            ]
        return []

    # ---------- 人数突变 ----------
    def _rule_spike(self, snapshot: ClassroomSnapshot, now: float) -> list[RuleOutcome]:
        rule_id = "occupancy_spike"
        if not self._enabled(rule_id):
            return []
        delta = int(self._param(rule_id, "delta", 8))
        if snapshot.occupancy_spike >= delta and self._cooldown(rule_id).ready(now):
            self._cooldown(rule_id).fire(now)
            return [
                RuleOutcome(
                    kind="trigger",
                    rule_id=rule_id,
                    event_type=rule_id,
                    risk_level=self._risk(rule_id, "low"),
                    title=EVENT_TYPE_LABELS[rule_id],
                    description=(
                        f"最近 {self._param(rule_id, 'window_seconds', 10)} 秒内人数变化 "
                        f"{snapshot.occupancy_spike} 人。"
                    ),
                    basis={"spike": snapshot.occupancy_spike, "threshold": delta},
                    needs_qwen=True,
                    source="fusion",
                )
            ]
        return []

    # ---------- 多摄像头矛盾 ----------
    def _rule_conflict(self, snapshot: ClassroomSnapshot, now: float) -> list[RuleOutcome]:
        rule_id = "camera_count_conflict"
        if not self._enabled(rule_id):
            return []
        duration = float(self._param(rule_id, "duration_seconds", 20.0))
        outcomes: list[RuleOutcome] = []
        active_regions = set()
        for conflict in snapshot.conflicts:
            region = conflict.get("region", "")
            active_regions.add(region)
            since = self._conflict_since.setdefault(region, now)
            cooldown = self._cooldown(rule_id, region)
            if now - since >= duration and cooldown.ready(now):
                cooldown.fire(now)
                outcomes.append(
                    RuleOutcome(
                        kind="trigger",
                        rule_id=rule_id,
                        event_type=rule_id,
                        risk_level=self._risk(rule_id, "low"),
                        title=EVENT_TYPE_LABELS[rule_id],
                        description=(
                            f"区域「{region}」主副摄像头人数差异 {conflict.get('delta')} 人，"
                            f"持续超过 {duration:.0f} 秒。"
                        ),
                        basis=conflict,
                        camera_id=conflict.get("primary_camera"),
                        needs_qwen=True,
                        source="fusion",
                    )
                )
        for region in list(self._conflict_since):
            if region not in active_regions:
                self._conflict_since.pop(region, None)
        return outcomes

    # ---------- 摄像头遮挡 ----------
    def _rule_obstruction(self, snapshot: ClassroomSnapshot, now: float) -> list[RuleOutcome]:
        rule_id = "camera_obstructed"
        if not self._enabled(rule_id):
            return []
        duration = float(self._param(rule_id, "duration_seconds", 15.0))
        outcomes: list[RuleOutcome] = []
        for cam in snapshot.cameras:
            if cam.obstructed and cam.online:
                since = self._camera_obstruction_since.setdefault(cam.camera_id, now)
                state_key = f"{rule_id}:{cam.camera_id}"
                cooldown = self._cooldown(rule_id, cam.camera_id)
                if now - since >= duration and cooldown.ready(now):
                    cooldown.fire(now)
                    outcomes.append(
                        RuleOutcome(
                            kind="trigger",
                            rule_id=rule_id,
                            event_type=rule_id,
                            risk_level=self._risk(rule_id, "medium"),
                            title=EVENT_TYPE_LABELS[rule_id],
                            description=f"{cam.camera_id} 画面持续异常，疑似被遮挡或失焦。",
                            basis={"camera_id": cam.camera_id, "duration_seconds": duration,
                                   "state_key": state_key},
                            camera_id=cam.camera_id,
                            needs_qwen=True,
                            source="vision",
                        )
                    )
            else:
                self._camera_obstruction_since.pop(cam.camera_id, None)
        return outcomes

    # ---------- 设备离线 ----------
    def _rule_device_offline(self, snapshot: ClassroomSnapshot, now: float) -> list[RuleOutcome]:
        rule_id = "device_offline"
        if not self._enabled(rule_id):
            return []
        timeout = float(self._param(rule_id, "timeout_seconds", 15.0))
        outcomes: list[RuleOutcome] = []
        offline = list(snapshot.offline_devices) + list(snapshot.offline_cameras)
        for device_id in offline:
            since = self._offline_since.setdefault(device_id, now)
            cooldown = self._cooldown(rule_id, device_id)
            if now - since >= timeout and cooldown.ready(now):
                cooldown.fire(now)
                outcomes.append(
                    RuleOutcome(
                        kind="trigger",
                        rule_id=rule_id,
                        event_type=rule_id,
                        risk_level=self._risk(rule_id, "medium"),
                        title=EVENT_TYPE_LABELS[rule_id],
                        description=f"设备 {device_id} 心跳超时，已标记为离线。",
                        basis={"device_id": device_id, "timeout_seconds": timeout},
                        camera_id=device_id if device_id.startswith("CAM-") else None,
                        needs_qwen=False,
                        source="system",
                    )
                )
        for device_id in list(self._offline_since):
            if device_id not in offline:
                self._offline_since.pop(device_id, None)
        return outcomes

    # ================= 状态汇总 =================
    def safety_state(self) -> tuple[SafetyState, str]:
        """当前教室安全状态 = 所有活跃规则中的最高等级。"""
        best = SafetyState.NORMAL
        reason = "所有指标正常"
        for rule_id, state in self._states.items():
            if not state.active:
                continue
            rule = self._rules.get(rule_id, {})
            if rule.get("event_type") in HEALTH_ONLY_EVENTS:
                continue
            level = RISK_TO_SAFETY.get(self._risk(rule_id), SafetyState.NOTICE)
            if level.severity > best.severity:
                best = level
                reason = self._active_reasons.get(rule_id, rule.get("name", rule_id))
        self._last_safety = best
        return best, reason

    def active_rules(self) -> list[str]:
        return [rule_id for rule_id, state in self._states.items() if state.active]

    def state_of(self, rule_id: str) -> RuleState | None:
        return self._states.get(rule_id)

    def mark_event(self, rule_id: str, event_id: str) -> None:
        state = self._states.get(rule_id)
        if state:
            state.event_id = event_id

    def snapshot(self) -> dict[str, Any]:
        safety, reason = self.safety_state()
        return {
            "safety_state": safety.value,
            "reason": reason,
            "presence": self._presence_active,
            "active_rules": self.active_rules(),
            "rule_states": {k: v.label for k, v in self._states.items()},
        }
