"""ESP-NOW 网关串口协议：换行分隔的 JSON 消息。

消息类型：telemetry / heartbeat / command / ack / event / log
本模块只负责编解码与序号统计，不涉及 IO。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

MAX_LINE_BYTES = 4096


class ProtocolError(ValueError):
    """JSON 格式错误或必填字段缺失。"""


def parse_line(line: str) -> dict[str, Any]:
    line = line.strip()
    if not line:
        raise ProtocolError("空行")
    if len(line.encode("utf-8", errors="ignore")) > MAX_LINE_BYTES:
        raise ProtocolError("单行长度超限")
    try:
        payload = json.loads(line)
    except (ValueError, TypeError) as exc:
        raise ProtocolError(f"JSON 解析失败: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("消息必须是 JSON 对象")
    msg_type = payload.get("type")
    if not msg_type:
        raise ProtocolError("缺少 type 字段")
    if msg_type in {"telemetry", "heartbeat"} and not payload.get("node_id"):
        raise ProtocolError("缺少 node_id 字段")
    if msg_type == "ack" and not payload.get("request_id"):
        raise ProtocolError("ack 缺少 request_id")
    return payload


def encode_command(
    request_id: str, target: str, action: str, data: dict[str, Any] | None = None
) -> bytes:
    payload = {
        "type": "command",
        "request_id": request_id,
        "target": target,
        "action": action,
        "timestamp": int(time.time()),
        "data": data or {},
    }
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


@dataclass
class SeqTracker:
    """按节点统计序号跳变、重复包与丢包率。"""

    node_id: str
    last_seq: int | None = None
    received: int = 0
    lost: int = 0
    duplicated: int = 0
    out_of_order: int = 0
    first_seq: int | None = None
    recent: list[int] = field(default_factory=list)

    def observe(self, seq: int | None) -> str:
        """返回 'ok' / 'duplicate' / 'gap' / 'reset'。"""
        self.received += 1
        if seq is None:
            return "ok"
        if self.first_seq is None:
            self.first_seq = seq
            self.last_seq = seq
            self.recent = [seq]
            return "ok"

        result = "ok"
        if seq in self.recent:
            self.duplicated += 1
            result = "duplicate"
        elif self.last_seq is not None and seq < self.last_seq:
            # 节点重启导致序号回绕（序号大幅回退，或直接从头开始计数）
            if self.last_seq - seq > 100 or seq <= 3:
                self.lost = 0
                self.first_seq = seq
                result = "reset"
            else:
                self.out_of_order += 1
                result = "gap"
        elif self.last_seq is not None and seq > self.last_seq + 1:
            self.lost += seq - self.last_seq - 1
            result = "gap"

        if result != "duplicate":
            self.last_seq = max(seq, self.last_seq or seq)
        self.recent.append(seq)
        if len(self.recent) > 32:
            self.recent = self.recent[-32:]
        return result

    @property
    def packet_rate(self) -> float:
        expected = self.received + self.lost
        if expected <= 0:
            return 1.0
        return max(0.0, min(1.0, self.received / expected))

    def reset(self) -> None:
        self.last_seq = None
        self.first_seq = None
        self.received = 0
        self.lost = 0
        self.duplicated = 0
        self.out_of_order = 0
        self.recent = []
