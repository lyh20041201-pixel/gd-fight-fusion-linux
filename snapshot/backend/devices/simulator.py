"""模拟设备通道。

在没有 ESP32 硬件时，生成与真实网关完全一致的换行 JSON 消息流：
温湿度 / CO₂ / 光照 / 噪声 / 烟感 / 报警节点心跳与 ACK，
并可注入掉线、丢包、延迟与各类风险场景。
"""

from __future__ import annotations

import json
import logging
import math
import random
import threading
import time
from typing import Any

from .base import DeviceAdapter
from .protocol import ProtocolError, parse_line

logger = logging.getLogger(__name__)

NODE_ENVIRONMENT = "environment_01"
NODE_NOISE = "noise_01"
NODE_SMOKE = "smoke_01"
NODE_ALARM = "alarm_01"
NODE_GATEWAY = "gateway_01"

SIM_NODES = [NODE_GATEWAY, NODE_ENVIRONMENT, NODE_NOISE, NODE_SMOKE, NODE_ALARM]


class _NodeSim:
    def __init__(self, node_id: str, interval: float) -> None:
        self.node_id = node_id
        self.interval = interval
        self.seq = random.randint(1000, 2000)
        self.next_at = time.time() + random.uniform(0, interval)
        self.online = True
        self.rssi = random.randint(-70, -45)
        self.offline_until: float | None = None
        self.extra_delay = 0.0


class SimulatedGateway(DeviceAdapter):
    """模拟 ESP-NOW 网关（线程驱动，行为与串口通道一致）。"""

    name = "simulated_gateway"

    def __init__(self, *, loss_rate: float = 0.02, seed: int | None = None) -> None:
        super().__init__()
        self._rng = random.Random(seed)
        self._loss_rate = loss_rate
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._connected = False
        self._lock = threading.RLock()
        self._start_time = time.time()

        self._nodes = {
            NODE_GATEWAY: _NodeSim(NODE_GATEWAY, 5.0),
            NODE_ENVIRONMENT: _NodeSim(NODE_ENVIRONMENT, 2.0),
            NODE_NOISE: _NodeSim(NODE_NOISE, 1.0),
            NODE_SMOKE: _NodeSim(NODE_SMOKE, 5.0),
            NODE_ALARM: _NodeSim(NODE_ALARM, 5.0),
        }

        # 环境基线
        self._co2_base = 620.0
        self._co2_offset = 0.0
        self._noise_base = 48.0
        self._noise_offset = 0.0
        self._temp_base = 24.5
        self._humidity_base = 48.0
        self._light_base = 380.0
        self._smoke_active = False
        self._smoke_clear_at: float | None = None
        self._people_hint = 0

    # ---------- 生命周期 ----------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._connected = True
        self._thread = threading.Thread(
            target=self._run, name="sim-gateway", daemon=True
        )
        self._thread.start()
        logger.info("模拟设备网关已启动（不会向任何真实串口写入数据）")

    def stop(self) -> None:
        self._stop.set()
        self._connected = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("模拟设备网关已停止")

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def port_name(self) -> str:
        return "SIMULATED"

    @property
    def simulated(self) -> bool:
        return True

    # ---------- 下行 ----------
    def send(self, payload: bytes) -> bool:
        """模拟模式下命令不会到达任何真实硬件，仅回 ACK。"""
        try:
            message = parse_line(payload.decode("utf-8"))
        except (ProtocolError, UnicodeDecodeError) as exc:
            logger.warning("模拟网关收到非法命令: %s", exc)
            return False
        target = message.get("target", NODE_ALARM)
        request_id = message.get("request_id")
        node = self._nodes.get(target)
        if node is not None and not node.online:
            return True  # 命令写出成功，但节点离线，最终会 ACK 超时
        delay = self._rng.uniform(0.05, 0.25)
        timer = threading.Timer(
            delay,
            self._emit_ack,
            args=(request_id, target),
        )
        timer.daemon = True
        timer.start()
        return True

    def _emit_ack(self, request_id: str | None, target: str) -> None:
        if self._stop.is_set():
            return
        self.emit(
            {
                "type": "ack",
                "request_id": request_id,
                "node_id": target,
                "status": "ok",
                "timestamp": int(time.time()),
            }
        )

    # ---------- 场景注入 ----------
    def inject(self, scenario: str, node_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if scenario == "smoke_detected":
                self._smoke_active = True
                self._smoke_clear_at = time.time() + 60
                self._emit_smoke(alarm=True)
                return {"scenario": scenario, "node": NODE_SMOKE}
            if scenario == "co2_critical":
                self._co2_offset = 1400.0
                return {"scenario": scenario, "co2": self._co2_base + self._co2_offset}
            if scenario == "noise_abnormal":
                self._noise_offset = 35.0
                return {"scenario": scenario}
            if scenario == "node_offline":
                target = node_id or NODE_ENVIRONMENT
                node = self._nodes.get(target)
                if node:
                    node.online = False
                    node.offline_until = time.time() + 90
                return {"scenario": scenario, "node": target}
            if scenario == "recover":
                self._co2_offset = 0.0
                self._noise_offset = 0.0
                self._smoke_active = False
                self._smoke_clear_at = None
                for node in self._nodes.values():
                    node.online = True
                    node.offline_until = None
                self._emit_smoke(alarm=False)
                return {"scenario": scenario}
        return {"scenario": scenario, "ignored": True}

    def set_people_hint(self, count: int) -> None:
        """人数会轻微影响 CO₂ 与噪声，使模拟数据之间彼此自洽。"""
        self._people_hint = max(0, int(count))

    # ---------- 主循环 ----------
    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            for node in self._nodes.values():
                if node.offline_until and now >= node.offline_until:
                    node.online = True
                    node.offline_until = None
                if now < node.next_at:
                    continue
                node.next_at = now + node.interval
                if not node.online:
                    continue
                if self._rng.random() < self._loss_rate:
                    node.seq += 1  # 丢包：序号推进但消息不发出
                    continue
                if self._rng.random() < 0.03:
                    time.sleep(self._rng.uniform(0.1, 0.4))  # 模拟数据延迟
                self._emit_node(node, now)
            self._maybe_clear_smoke(now)
            self._stop.wait(0.2)

    def _emit_node(self, node: _NodeSim, now: float) -> None:
        node.seq += 1
        node.rssi = max(-92, min(-38, node.rssi + self._rng.randint(-3, 3)))
        base = {
            "node_id": node.node_id,
            "seq": node.seq,
            "timestamp": int(now),
            "rssi": node.rssi,
        }
        if node.node_id == NODE_ENVIRONMENT:
            self.emit({"type": "telemetry", **base, "data": self._environment_data(now)})
        elif node.node_id == NODE_NOISE:
            self.emit({"type": "telemetry", **base, "data": {"noise": self._noise(now)}})
        elif node.node_id == NODE_SMOKE:
            self.emit(
                {
                    "type": "telemetry",
                    **base,
                    "data": {"smoke": 1 if self._smoke_active else 0},
                }
            )
        else:
            self.emit({"type": "heartbeat", **base})

    def _environment_data(self, now: float) -> dict[str, float]:
        phase = now / 240.0
        people = self._people_hint
        temperature = round(
            self._temp_base + 1.6 * math.sin(phase) + people * 0.03
            + self._rng.uniform(-0.15, 0.15),
            1,
        )
        humidity = round(
            self._humidity_base + 5.0 * math.sin(phase / 1.7) + people * 0.08
            + self._rng.uniform(-0.4, 0.4),
            1,
        )
        co2 = round(
            self._co2_base
            + self._co2_offset
            + 90 * math.sin(phase / 2.3)
            + people * 14.0
            + self._rng.uniform(-18, 18)
        )
        light = round(
            max(0.0, self._light_base + 120 * math.sin(phase / 3.1)
                + self._rng.uniform(-25, 25)),
            1,
        )
        # 注入的高浓度会缓慢回落，配合规则的滞回解除阈值
        if self._co2_offset > 0:
            self._co2_offset = max(0.0, self._co2_offset - 6.0)
        return {
            "temperature": temperature,
            "humidity": humidity,
            "co2": float(co2),
            "light": light,
        }

    def _noise(self, now: float) -> float:
        value = (
            self._noise_base
            + self._noise_offset
            + 6.0 * math.sin(now / 30.0)
            + self._people_hint * 0.35
            + self._rng.uniform(-2.5, 3.5)
        )
        if self._noise_offset > 0:
            self._noise_offset = max(0.0, self._noise_offset - 0.15)
        return round(max(28.0, min(110.0, value)), 1)

    def _emit_smoke(self, alarm: bool) -> None:
        node = self._nodes[NODE_SMOKE]
        node.seq += 1
        self.emit(
            {
                "type": "telemetry",
                "node_id": NODE_SMOKE,
                "seq": node.seq,
                "timestamp": int(time.time()),
                "rssi": node.rssi,
                "data": {"smoke": 1 if alarm else 0},
            }
        )

    def _maybe_clear_smoke(self, now: float) -> None:
        if self._smoke_active and self._smoke_clear_at and now >= self._smoke_clear_at:
            self._smoke_active = False
            self._smoke_clear_at = None
            self._emit_smoke(alarm=False)

    # 便于测试
    def feed_raw(self, line: str) -> None:
        try:
            self.emit(parse_line(line))
        except ProtocolError as exc:
            self.emit_error(str(exc))

    def snapshot(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(
                {
                    "co2_offset": self._co2_offset,
                    "noise_offset": self._noise_offset,
                    "smoke": self._smoke_active,
                    "nodes": {n: s.online for n, s in self._nodes.items()},
                }
            )
        )
