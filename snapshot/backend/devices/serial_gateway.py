"""真实串口网关适配器（ESP32-C3 USB 串口）。

- 自动发现或按配置打开串口
- 断线自动重连（指数退避上限固定）
- 按行读取，单行 JSON 解析失败只记一次错误，不影响后续数据
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .base import DeviceAdapter
from .protocol import ProtocolError, parse_line

logger = logging.getLogger(__name__)

# 常见 ESP32-C3 板载 USB 转串口芯片
KNOWN_VID_PID = {
    (0x303A, 0x1001),  # Espressif USB JTAG/serial
    (0x1A86, 0x7523),  # CH340
    (0x1A86, 0x55D4),  # CH9102
    (0x10C4, 0xEA60),  # CP2102
    (0x0403, 0x6001),  # FT232
}
KEYWORDS = ("esp32", "ch340", "ch910", "cp210", "usb-serial", "usb serial", "silicon labs")


def list_serial_ports() -> list[dict[str, str]]:
    try:
        from serial.tools import list_ports
    except ImportError:  # pragma: no cover
        return []
    ports = []
    for port in list_ports.comports():
        ports.append(
            {
                "device": port.device,
                "description": port.description or "",
                "hwid": port.hwid or "",
                "manufacturer": port.manufacturer or "",
            }
        )
    return ports


def discover_port() -> str | None:
    try:
        from serial.tools import list_ports
    except ImportError:  # pragma: no cover
        return None
    candidates = list(list_ports.comports())
    for port in candidates:
        if port.vid is not None and (port.vid, port.pid) in KNOWN_VID_PID:
            return port.device
    for port in candidates:
        text = f"{port.description or ''} {port.manufacturer or ''}".lower()
        if any(k in text for k in KEYWORDS):
            return port.device
    return candidates[0].device if candidates else None


class SerialGateway(DeviceAdapter):
    name = "serial_gateway"

    def __init__(
        self,
        port: str = "",
        baudrate: int = 115200,
        *,
        auto_discover: bool = True,
        reconnect_seconds: float = 3.0,
    ) -> None:
        super().__init__()
        self._configured_port = port
        self._port = port
        self._baudrate = baudrate
        self._auto_discover = auto_discover
        self._reconnect_seconds = reconnect_seconds
        self._serial: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._connected = False
        self._last_error: str | None = None
        self._parse_errors = 0
        self._reconnects = 0

    # ---------- 生命周期 ----------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="serial-gateway", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:  # noqa: BLE001
                    pass
                self._serial = None
            self._connected = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("串口网关已关闭")

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def port_name(self) -> str:
        return self._port or self._configured_port or "AUTO"

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def stats(self) -> dict[str, Any]:
        return {
            "port": self.port_name,
            "connected": self._connected,
            "parse_errors": self._parse_errors,
            "reconnects": self._reconnects,
            "last_error": self._last_error,
        }

    # ---------- 下行 ----------
    def send(self, payload: bytes) -> bool:
        with self._lock:
            if self._serial is None or not self._connected:
                self._last_error = "串口未连接，命令未发送"
                return False
            try:
                self._serial.write(payload)
                self._serial.flush()
                return True
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"串口写入失败: {exc}"
                logger.error(self._last_error)
                self._drop_connection()
                return False

    # ---------- 主循环 ----------
    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._connected and not self._open():
                self._stop.wait(self._reconnect_seconds)
                continue
            self._read_loop()

    def _open(self) -> bool:
        try:
            import serial
        except ImportError:  # pragma: no cover
            self._last_error = "缺少 pyserial 依赖"
            self.emit_error(self._last_error)
            return False

        port = self._configured_port or (discover_port() if self._auto_discover else "")
        if not port:
            self._last_error = "未找到可用串口"
            self.emit_error(self._last_error)
            return False
        try:
            handle = serial.Serial(port=port, baudrate=self._baudrate, timeout=1.0)
        except Exception as exc:  # noqa: BLE001 - pyserial 会抛多种异常
            self._last_error = f"打开串口 {port} 失败: {exc}"
            self.emit_error(self._last_error)
            return False

        with self._lock:
            self._serial = handle
            self._port = port
            self._connected = True
            self._last_error = None
        self._reconnects += 1
        logger.info("串口已连接: %s @ %d", port, self._baudrate)
        self.emit({"type": "gateway_connected", "port": port})
        return True

    def _read_loop(self) -> None:
        buffer = b""
        while not self._stop.is_set() and self._connected:
            try:
                chunk = self._serial.readline()
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"串口读取失败: {exc}"
                logger.warning(self._last_error)
                self._drop_connection()
                return
            if not chunk:
                continue
            buffer += chunk
            if not buffer.endswith(b"\n"):
                if len(buffer) > 8192:
                    buffer = b""
                continue
            for raw in buffer.split(b"\n"):
                if not raw.strip():
                    continue
                self._handle_raw(raw)
            buffer = b""

    def _handle_raw(self, raw: bytes) -> None:
        try:
            text = raw.decode("utf-8", errors="replace")
            message = parse_line(text)
        except ProtocolError as exc:
            self._parse_errors += 1
            if self._parse_errors <= 5 or self._parse_errors % 50 == 0:
                logger.warning("串口消息解析失败(%d): %s", self._parse_errors, exc)
            return
        self.emit(message)

    def _drop_connection(self) -> None:
        with self._lock:
            self._connected = False
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:  # noqa: BLE001
                    pass
                self._serial = None
        self.emit({"type": "gateway_disconnected", "reason": self._last_error or "unknown"})
        time.sleep(0.2)
