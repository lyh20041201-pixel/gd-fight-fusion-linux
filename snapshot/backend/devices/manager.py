"""设备管理：节点注册、消息处理、心跳超时、命令下发与 ACK 重试。

模拟适配器与真实串口适配器都经过这里，业务层看到的接口完全一致。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config.runtime import RuntimeConfig
from ..schemas.common import DeviceOut, ModuleStatus, NodeStats
from ..schemas.enums import ModuleState, ModuleType, SensorKind
from .base import DeviceAdapter
from .protocol import SeqTracker, encode_command
from .validation import (
    DEFAULT_STALE_SECONDS,
    finite_only,
    is_stale,
    validate_reading,
)

logger = logging.getLogger(__name__)

SENSOR_META: dict[str, tuple[str, str]] = {
    "temperature": ("温度", "°C"),
    "humidity": ("湿度", "%"),
    "co2": ("CO₂ 浓度", "ppm"),
    "light": ("光照", "lux"),
    "noise": ("噪声", "dB"),
    "smoke": ("烟雾报警", ""),
}


# 最近多少秒内出现过无效读数才在界面上提示
INVALID_NOTICE_SECONDS = 60.0

# 模块状态的严重程度排序，用于取"更严重的那个"
_MODULE_STATE_ORDER: dict[ModuleState, int] = {
    ModuleState.INITIALIZING: 0,
    ModuleState.ONLINE: 1,
    ModuleState.OFFLINE: 2,
    ModuleState.WARNING: 3,
    ModuleState.ALARM: 4,
    ModuleState.FAULT: 5,
}


@dataclass
class NodeDefinition:
    node_id: str
    name: str
    module_type: ModuleType
    sensors: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    critical: bool = True


DEFAULT_NODES: list[NodeDefinition] = [
    NodeDefinition("gateway_01", "ESP-NOW 网关", ModuleType.GATEWAY, [], ["relay"]),
    NodeDefinition(
        "environment_01",
        "环境采集节点",
        ModuleType.NODE,
        ["temperature", "humidity", "co2", "light"],
    ),
    NodeDefinition("noise_01", "噪声采集节点 (INMP441)", ModuleType.NODE, ["noise"]),
    NodeDefinition("smoke_01", "光电烟感节点 (JTY-GD-301AL)", ModuleType.NODE, ["smoke"]),
    NodeDefinition(
        "alarm_01",
        "声光报警节点",
        ModuleType.ACTUATOR,
        [],
        ["led", "buzzer", "oled"],
    ),
]


@dataclass
class NodeRuntime:
    definition: NodeDefinition
    tracker: SeqTracker
    online: bool = False
    rssi: int | None = None
    last_seen: float | None = None
    last_heartbeat: float | None = None
    values: dict[str, float] = field(default_factory=dict)
    value_times: dict[str, float] = field(default_factory=dict)
    fault_reason: str | None = None
    # 无效读数统计：{传感器种类: 次数} 与最近一次的原因，用于界面上说明"为什么没数据"
    invalid_counts: dict[str, int] = field(default_factory=dict)
    invalid_reasons: dict[str, str] = field(default_factory=dict)
    invalid_times: dict[str, float] = field(default_factory=dict)


@dataclass
class PendingCommand:
    request_id: str
    target: str
    action: str
    data: dict[str, Any]
    sent_at: float
    retries: int = 0
    future: asyncio.Future | None = None


class DeviceManager:
    def __init__(
        self,
        adapter: DeviceAdapter,
        runtime: RuntimeConfig,
        *,
        ack_timeout: float = 2.0,
        max_retries: int = 3,
        heartbeat_timeout: float = 15.0,
    ) -> None:
        self.adapter = adapter
        self.runtime = runtime
        self.ack_timeout = ack_timeout
        self.max_retries = max_retries
        self.heartbeat_timeout = heartbeat_timeout

        self._lock = threading.RLock()
        self._nodes: dict[str, NodeRuntime] = {
            d.node_id: NodeRuntime(definition=d, tracker=SeqTracker(d.node_id))
            for d in DEFAULT_NODES
        }
        self._pending: dict[str, PendingCommand] = {}
        self._command_seq = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._monitor_task: asyncio.Task | None = None
        self._started_at = time.time()
        self._gateway_error: str | None = None
        self._parse_errors = 0

        # 回调（由 AppState 注入）
        self.on_sensor_update: Callable[[str, str, float, float], None] | None = None
        self.on_device_change: Callable[[str], None] | None = None
        self.on_critical_change: Callable[[str, float], None] | None = None

        adapter.set_handler(self._handle_message)
        adapter.set_error_handler(self._handle_channel_error)

    # ================= 生命周期 =================
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.adapter.start()
        self._monitor_task = asyncio.create_task(self._monitor_loop(), name="device-monitor")
        logger.info("设备管理器已启动，通道=%s", self.adapter.name)

    async def stop(self) -> None:
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        await asyncio.to_thread(self.adapter.stop)
        logger.info("设备管理器已停止")

    # ================= 上行消息 =================
    def _handle_message(self, message: dict[str, Any]) -> None:
        msg_type = message.get("type")
        if msg_type == "telemetry":
            self._handle_telemetry(message)
        elif msg_type == "heartbeat":
            self._handle_heartbeat(message)
        elif msg_type == "ack":
            self._handle_ack(message)
        elif msg_type == "gateway_connected":
            self._gateway_error = None
            self._notify_device("gateway_01")
        elif msg_type == "gateway_disconnected":
            self._gateway_error = str(message.get("reason", "串口断开"))
            with self._lock:
                for node in self._nodes.values():
                    node.online = False
                    node.fault_reason = "网关断开"
            self._notify_device("gateway_01")
        elif msg_type == "log":
            logger.info("节点日志 %s: %s", message.get("node_id"), message.get("message"))

    def _touch_node(self, node_id: str, message: dict[str, Any]) -> NodeRuntime | None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                definition = NodeDefinition(node_id, node_id, ModuleType.NODE, [])
                node = NodeRuntime(definition=definition, tracker=SeqTracker(node_id))
                self._nodes[node_id] = node
                logger.info("发现新节点: %s", node_id)
            seq = message.get("seq")
            result = node.tracker.observe(int(seq) if seq is not None else None)
            if result == "duplicate":
                return None
            node.online = True
            node.fault_reason = None
            node.rssi = message.get("rssi", node.rssi)
            node.last_seen = time.time()
            # 网关本身在收到任意节点消息时视为在线
            gw = self._nodes.get("gateway_01")
            if gw is not None and node_id != "gateway_01":
                gw.online = True
                gw.last_seen = node.last_seen
                gw.fault_reason = None
            return node

    def _handle_telemetry(self, message: dict[str, Any]) -> None:
        node_id = str(message.get("node_id"))
        node = self._touch_node(node_id, message)
        if node is None:
            return
        data = message.get("data") or {}
        ts = time.time()
        updates: list[tuple[str, float]] = []
        rejected: list[tuple[str, str]] = []
        with self._lock:
            for key, value in data.items():
                result = validate_reading(key, value)
                if result.invalid:
                    # 无效读数一律丢弃：不写入 values，也不刷新 value_times。
                    # 该传感器随后会因为读数陈旧而显示"无数据"，绝不用 0 顶替。
                    node.invalid_counts[key] = node.invalid_counts.get(key, 0) + 1
                    node.invalid_reasons[key] = result.reason or "无效读数"
                    node.invalid_times[key] = ts
                    rejected.append((key, result.reason or "无效读数"))
                    continue
                number = float(result.value or 0.0)
                previous = node.values.get(key)
                node.values[key] = number
                node.value_times[key] = ts
                updates.append((key, number))
                if key == "smoke" and previous != number and number >= 1:
                    self._fire_critical(node_id, number)
        for key, reason in rejected:
            logger.warning("丢弃无效读数 %s.%s: %s", node_id, key, reason)
        for key, value in updates:
            if self.on_sensor_update:
                try:
                    self.on_sensor_update(node_id, key, value, ts)
                except Exception:  # noqa: BLE001
                    logger.exception("传感器回调失败 %s.%s", node_id, key)
        self._notify_device(node_id)

    def _handle_heartbeat(self, message: dict[str, Any]) -> None:
        node_id = str(message.get("node_id"))
        node = self._touch_node(node_id, message)
        if node is None:
            return
        with self._lock:
            node.last_heartbeat = time.time()
        self._notify_device(node_id)

    def _handle_ack(self, message: dict[str, Any]) -> None:
        request_id = str(message.get("request_id"))
        status = message.get("status", "ok")
        with self._lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            return
        if pending.future is not None and not pending.future.done():
            loop = self._loop
            if loop is not None and not loop.is_closed():
                loop.call_soon_threadsafe(
                    lambda: (
                        pending.future.set_result(status)
                        if pending.future and not pending.future.done()
                        else None
                    )
                )

    def _handle_channel_error(self, reason: str) -> None:
        self._gateway_error = reason
        self._notify_device("gateway_01")

    def _fire_critical(self, node_id: str, value: float) -> None:
        if not self.on_critical_change:
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(lambda: self.on_critical_change(node_id, value))  # type: ignore[misc]

    def _notify_device(self, node_id: str) -> None:
        if not self.on_device_change:
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(lambda: self.on_device_change(node_id))  # type: ignore[misc]

    # ================= 心跳监控 =================
    async def _monitor_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(2.0)
                self._check_timeouts()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("设备心跳监控异常")

    def _check_timeouts(self) -> None:
        timeout = float(
            self.runtime.threshold("heartbeat_timeout_seconds", self.heartbeat_timeout)
        )
        now = time.time()
        changed: list[str] = []
        with self._lock:
            for node_id, node in self._nodes.items():
                reference = max(
                    [t for t in (node.last_seen, node.last_heartbeat) if t] or [0.0]
                )
                if reference <= 0:
                    continue
                if node.online and now - reference > timeout:
                    node.online = False
                    node.fault_reason = f"心跳超时 {now - reference:.0f}s"
                    changed.append(node_id)
        for node_id in changed:
            logger.warning("节点 %s 心跳超时，标记为离线", node_id)
            if self.on_device_change:
                self.on_device_change(node_id)

    # ================= 下行命令 =================
    def _next_request_id(self) -> str:
        with self._lock:
            self._command_seq += 1
            return f"cmd_{self._command_seq:05d}"

    async def send_command(
        self,
        target: str,
        action: str,
        data: dict[str, Any] | None = None,
        *,
        wait_ack: bool = True,
    ) -> dict[str, Any]:
        """发送命令并等待 ACK，超时自动重试有限次。"""
        request_id = self._next_request_id()
        payload = encode_command(request_id, target, action, data)
        loop = asyncio.get_running_loop()
        attempts = 0
        last_error = ""

        while attempts <= self.max_retries:
            attempts += 1
            future: asyncio.Future = loop.create_future()
            pending = PendingCommand(
                request_id=request_id,
                target=target,
                action=action,
                data=data or {},
                sent_at=time.time(),
                retries=attempts - 1,
                future=future,
            )
            with self._lock:
                self._pending[request_id] = pending

            ok = await asyncio.to_thread(self.adapter.send, payload)
            if not ok:
                with self._lock:
                    self._pending.pop(request_id, None)
                last_error = "通道不可用"
                await asyncio.sleep(0.2)
                continue
            if not wait_ack:
                with self._lock:
                    self._pending.pop(request_id, None)
                return {
                    "ok": True,
                    "request_id": request_id,
                    "status": "sent",
                    "retries": attempts - 1,
                }
            try:
                status = await asyncio.wait_for(future, timeout=self.ack_timeout)
                return {
                    "ok": status == "ok",
                    "request_id": request_id,
                    "status": status,
                    "retries": attempts - 1,
                }
            except asyncio.TimeoutError:
                last_error = "ACK 超时"
                with self._lock:
                    self._pending.pop(request_id, None)
                logger.warning(
                    "命令 %s -> %s 第 %d 次 ACK 超时", action, target, attempts
                )

        return {
            "ok": False,
            "request_id": request_id,
            "status": "timeout",
            "error": last_error,
            "retries": attempts - 1,
        }

    # ================= 查询 =================
    def latest_values(self) -> dict[str, float]:
        """当前有效的传感器读数。

        离线节点、陈旧读数（超过 stale 窗口没有新数据）都不会出现在结果里 ——
        规则引擎看到"没有这个键"而不是"看到一个过期的值"，
        避免用几分钟前的数据继续做安全判定。
        """
        out: dict[str, float] = {}
        now = time.time()
        with self._lock:
            for node in self._nodes.values():
                if not node.online:
                    continue
                for kind, value in node.values.items():
                    if is_stale(node.value_times.get(kind), now, DEFAULT_STALE_SECONDS):
                        continue
                    out[kind] = value
        return finite_only(out)

    def sensor_online(self, kind: str) -> bool:
        with self._lock:
            for node in self._nodes.values():
                if kind in node.definition.sensors:
                    return node.online
        return False

    def node_stats(self) -> list[NodeStats]:
        with self._lock:
            return [
                NodeStats(
                    node_id=node.definition.node_id,
                    name=node.definition.name,
                    online=node.online,
                    rssi=node.rssi,
                    last_seen=node.last_seen,
                    last_seq=node.tracker.last_seq,
                    received=node.tracker.received,
                    lost=node.tracker.lost,
                    duplicated=node.tracker.duplicated,
                    packet_rate=round(node.tracker.packet_rate, 4),
                    values=dict(node.values),
                )
                for node in self._nodes.values()
            ]

    def devices(self) -> list[DeviceOut]:
        out: list[DeviceOut] = []
        with self._lock:
            for node in self._nodes.values():
                definition = node.definition
                actions = ["reconnect"]
                if "led" in definition.capabilities:
                    actions.append("test_led")
                if "buzzer" in definition.capabilities:
                    actions.append("test_buzzer")
                state = ModuleState.ONLINE if node.online else ModuleState.OFFLINE
                if definition.node_id == "gateway_01" and not self.adapter.connected:
                    state = ModuleState.FAULT
                out.append(
                    DeviceOut(
                        device_id=definition.node_id,
                        name=definition.name,
                        device_type=definition.module_type,
                        state=state,
                        online=node.online,
                        rssi=node.rssi,
                        packet_rate=round(node.tracker.packet_rate, 4),
                        last_seen=node.last_seen,
                        last_heartbeat=node.last_heartbeat,
                        values=dict(node.values),
                        fault_reason=node.fault_reason
                        or (self._gateway_error if definition.node_id == "gateway_01" else None),
                        actions=actions,
                    )
                )
        return out

    def module_statuses(self) -> list[ModuleStatus]:
        """网关 / 节点 / 每个传感器 / 执行器 的模块状态。"""
        modules: list[ModuleStatus] = []
        thresholds = self.runtime.thresholds
        now = time.time()
        with self._lock:
            for node in self._nodes.values():
                definition = node.definition
                if definition.node_id == "gateway_01":
                    connected = self.adapter.connected
                    modules.append(
                        ModuleStatus(
                            module_id="gateway_01",
                            name=definition.name,
                            module_type=ModuleType.GATEWAY,
                            state=ModuleState.ONLINE if connected else ModuleState.FAULT,
                            value_text=(
                                (
                                    f"{self.adapter.port_name} · "
                                    f"{'模拟通道' if self.adapter.simulated else '串口'}"
                                )
                                if connected
                                else "未连接"
                            ),
                            detail={
                                "port": self.adapter.port_name,
                                "simulated": self.adapter.simulated,
                                "nodes": len(self._nodes),
                            },
                            last_update=node.last_seen,
                            rssi=node.rssi,
                            packet_rate=round(node.tracker.packet_rate, 4),
                            fault_reason=None if connected else (self._gateway_error or "串口未连接"),
                            actions=["reconnect"],
                        )
                    )
                    continue

                node_state = ModuleState.ONLINE if node.online else ModuleState.OFFLINE
                value_text = "在线" if node.online else "离线"
                if not node.online and not node.fault_reason:
                    # 离线必须给出原因，界面上不能出现"离线但无故障说明"
                    node.fault_reason = (
                        "心跳超时" if node.last_seen else "尚未收到该节点任何数据"
                    )
                if node.online and node.tracker.packet_rate < 0.85:
                    node_state = ModuleState.WARNING
                    value_text = f"丢包率偏高 {(1 - node.tracker.packet_rate) * 100:.1f}%"
                actions = ["reconnect"]
                if "led" in definition.capabilities:
                    actions.append("test_led")
                if "buzzer" in definition.capabilities:
                    actions.append("test_buzzer")
                modules.append(
                    ModuleStatus(
                        module_id=definition.node_id,
                        name=definition.name,
                        module_type=definition.module_type,
                        state=node_state,
                        value_text=value_text,
                        detail={
                            "received": node.tracker.received,
                            "lost": node.tracker.lost,
                            "duplicated": node.tracker.duplicated,
                            "last_seq": node.tracker.last_seq,
                            "capabilities": definition.capabilities,
                        },
                        last_update=node.last_seen,
                        rssi=node.rssi,
                        packet_rate=round(node.tracker.packet_rate, 4),
                        fault_reason=node.fault_reason,
                        actions=actions,
                    )
                )

                # 每个传感器单独一个模块（数值异常 != 传感器离线）
                for kind in definition.sensors:
                    label, unit = SENSOR_META.get(kind, (kind, ""))
                    value = node.values.get(kind)
                    updated = node.value_times.get(kind)
                    stale = value is not None and is_stale(
                        updated, now, DEFAULT_STALE_SECONDS
                    )
                    invalid_count = node.invalid_counts.get(kind, 0)
                    invalid_reason = node.invalid_reasons.get(kind)
                    # 只有"最近"出现过无效读数才提示，避免开机时一次抖动就一直黄着
                    invalid_recent = invalid_count > 0 and not is_stale(
                        node.invalid_times.get(kind), now, INVALID_NOTICE_SECONDS
                    )
                    if not node.online or value is None or stale:
                        state = ModuleState.OFFLINE
                        text = "无数据"
                    else:
                        state, text = self._sensor_state(kind, value, unit, thresholds)
                        if invalid_recent:
                            # 最近有无效读数但当前值有效：黄色提示，数值照常显示
                            state = max(
                                state, ModuleState.WARNING, key=_MODULE_STATE_ORDER.get
                            )

                    # 没数据时必须说清楚为什么：节点离线 / 读数被判无效 / 读数陈旧
                    if not node.online:
                        reason: str | None = "所属节点离线"
                    elif value is None and invalid_reason:
                        reason = f"读数无效已丢弃：{invalid_reason}"
                    elif stale:
                        reason = "读数陈旧，超过 30 秒未更新"
                    elif invalid_recent:
                        reason = f"累计丢弃 {invalid_count} 条无效读数：{invalid_reason}"
                    else:
                        reason = None

                    modules.append(
                        ModuleStatus(
                            module_id=f"{definition.node_id}.{kind}",
                            name=label,
                            module_type=ModuleType.SENSOR,
                            state=state,
                            value_text=text,
                            detail={
                                "node_id": definition.node_id,
                                "kind": kind,
                                "unit": unit,
                                "invalid_readings": invalid_count,
                                "last_invalid_reason": invalid_reason,
                                "stale": stale,
                            },
                            last_update=updated,
                            rssi=node.rssi,
                            packet_rate=round(node.tracker.packet_rate, 4),
                            fault_reason=reason,
                        )
                    )

                # 执行器（RGB / 蜂鸣器 / OLED）
                for cap in definition.capabilities:
                    cap_label = {"led": "RGB 警示灯", "buzzer": "有源蜂鸣器", "oled": "OLED 显示"}.get(
                        cap, cap
                    )
                    modules.append(
                        ModuleStatus(
                            module_id=f"{definition.node_id}.{cap}",
                            name=cap_label,
                            module_type=(
                                ModuleType.DISPLAY if cap == "oled" else ModuleType.ACTUATOR
                            ),
                            state=ModuleState.ONLINE if node.online else ModuleState.OFFLINE,
                            value_text="待命" if node.online else "离线",
                            detail={"node_id": definition.node_id, "capability": cap},
                            last_update=node.last_heartbeat or node.last_seen,
                            rssi=node.rssi,
                            fault_reason=node.fault_reason,
                            actions=["test_led"] if cap == "led" else (
                                ["test_buzzer"] if cap == "buzzer" else []
                            ),
                        )
                    )
        return modules

    @staticmethod
    def _sensor_state(
        kind: str, value: float, unit: str, thresholds: dict[str, Any]
    ) -> tuple[ModuleState, str]:
        text = f"{value:g} {unit}".strip()
        if kind == SensorKind.CO2.value:
            if value >= float(thresholds.get("co2_critical", 1800)):
                return ModuleState.ALARM, text
            if value >= float(thresholds.get("co2_warning", 1000)):
                return ModuleState.WARNING, text
        elif kind == SensorKind.NOISE.value:
            if value >= float(thresholds.get("noise_warning", 75)):
                return ModuleState.WARNING, text
        elif kind == SensorKind.SMOKE.value:
            if value >= 1:
                return ModuleState.ALARM, "烟雾报警触发"
            return ModuleState.ONLINE, "正常"
        elif kind == SensorKind.TEMPERATURE.value:
            if value >= float(thresholds.get("temperature_high", 32)) or value <= float(
                thresholds.get("temperature_low", 8)
            ):
                return ModuleState.WARNING, text
        elif kind == SensorKind.HUMIDITY.value:
            if value >= float(thresholds.get("humidity_high", 80)) or value <= float(
                thresholds.get("humidity_low", 20)
            ):
                return ModuleState.WARNING, text
        return ModuleState.ONLINE, text

    # ================= 维护动作 =================
    async def reconnect(self, device_id: str | None = None) -> dict[str, Any]:
        """重连通道；对节点而言相当于清空统计并等待下一次心跳。"""
        if device_id and device_id != "gateway_01":
            with self._lock:
                node = self._nodes.get(device_id)
                if node is None:
                    return {"ok": False, "message": f"未知设备 {device_id}"}
                node.tracker.reset()
                node.fault_reason = None
            return {"ok": True, "message": f"已重置 {device_id} 统计，等待节点心跳"}

        await asyncio.to_thread(self.adapter.stop)
        await asyncio.sleep(0.3)
        await asyncio.to_thread(self.adapter.start)
        return {
            "ok": True,
            "message": f"通道 {self.adapter.name} 已重启",
            "detail": self.adapter.describe(),
        }

    def inject_scenario(self, scenario: str, node_id: str | None = None) -> dict[str, Any]:
        inject = getattr(self.adapter, "inject", None)
        if inject is None:
            return {"ok": False, "message": "当前通道不支持场景注入（真实模式）"}
        return {"ok": True, "detail": inject(scenario, node_id)}

    def set_people_hint(self, count: int) -> None:
        hint = getattr(self.adapter, "set_people_hint", None)
        if hint is not None:
            hint(count)

    def gateway_connected(self) -> bool:
        return self.adapter.connected

    def stats(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter.describe(),
            "pending_commands": len(self._pending),
            "uptime_seconds": time.time() - self._started_at,
            "gateway_error": self._gateway_error,
        }
