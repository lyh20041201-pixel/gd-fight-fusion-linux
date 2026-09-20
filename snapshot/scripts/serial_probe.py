"""串口排查工具：列出端口、监听网关原始报文、发送一条测试命令。

用法：
    python scripts/serial_probe.py list
    python scripts/serial_probe.py listen [COM5] [--seconds 20]
    python scripts/serial_probe.py send COM5 alarm_01 set_alarm '{"color":"green","buzzer":false}'

注意：send 会向真实硬件下发指令，请确认现场安全后再使用。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.devices.protocol import ProtocolError, encode_command, parse_line  # noqa: E402
from backend.devices.serial_gateway import discover_port, list_serial_ports  # noqa: E402


def cmd_list() -> int:
    ports = list_serial_ports()
    if not ports:
        print("未发现任何串口设备")
        return 1
    for port in ports:
        print(f"{port['device']:<10} {port['description']}  [{port['hwid']}]")
    guess = discover_port()
    print(f"\n自动发现结果: {guess or '无'}")
    return 0


def cmd_listen(port: str | None, seconds: float) -> int:
    import serial

    port = port or discover_port()
    if not port:
        print("未找到串口")
        return 1
    print(f"监听 {port} ... ({seconds:.0f}s)")
    stats: dict[str, int] = {}
    with serial.Serial(port, 115200, timeout=1.0) as handle:
        deadline = time.time() + seconds
        while time.time() < deadline:
            raw = handle.readline()
            if not raw:
                continue
            text = raw.decode("utf-8", errors="replace").strip()
            try:
                message = parse_line(text)
            except ProtocolError as exc:
                print(f"[解析失败] {exc}: {text[:120]}")
                stats["error"] = stats.get("error", 0) + 1
                continue
            key = f"{message.get('type')}/{message.get('node_id', '-')}"
            stats[key] = stats.get(key, 0) + 1
            print(json.dumps(message, ensure_ascii=False))
    print("\n统计:")
    for key, count in sorted(stats.items()):
        print(f"  {key}: {count}")
    return 0


def cmd_send(port: str, target: str, action: str, payload: str) -> int:
    import serial

    data = json.loads(payload) if payload else {}
    frame = encode_command("cmd_probe", target, action, data)
    with serial.Serial(port, 115200, timeout=2.0) as handle:
        handle.write(frame)
        handle.flush()
        print(f"已发送: {frame.decode().strip()}")
        deadline = time.time() + 3
        while time.time() < deadline:
            raw = handle.readline()
            if not raw:
                continue
            text = raw.decode("utf-8", errors="replace").strip()
            if '"ack"' in text:
                print(f"收到 ACK: {text}")
                return 0
    print("未收到 ACK")
    return 1


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] == "list":
        return cmd_list()
    if args[0] == "listen":
        port = args[1] if len(args) > 1 and not args[1].startswith("--") else None
        seconds = 20.0
        if "--seconds" in args:
            seconds = float(args[args.index("--seconds") + 1])
        return cmd_listen(port, seconds)
    if args[0] == "send" and len(args) >= 4:
        return cmd_send(args[1], args[2], args[3], args[4] if len(args) > 4 else "{}")
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
