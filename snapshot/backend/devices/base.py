"""设备适配器接口。

模拟适配器与真实串口适配器实现同一接口，模拟数据通过与真实设备
完全相同的通道进入系统（DeviceManager -> 状态/规则/WebSocket）。
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

MessageHandler = Callable[[dict[str, Any]], None]


class DeviceAdapter(abc.ABC):
    """设备通道抽象。"""

    name: str = "adapter"

    def __init__(self) -> None:
        self._handler: MessageHandler | None = None
        self._error_handler: Callable[[str], None] | None = None

    def set_handler(self, handler: MessageHandler) -> None:
        self._handler = handler

    def set_error_handler(self, handler: Callable[[str], None]) -> None:
        self._error_handler = handler

    def emit(self, message: dict[str, Any]) -> None:
        if self._handler is None:
            return
        try:
            self._handler(message)
        except Exception:  # noqa: BLE001 - 单条消息处理失败不得终止通道
            logger.exception("处理设备消息失败: %s", message.get("type"))

    def emit_error(self, reason: str) -> None:
        if self._error_handler is not None:
            try:
                self._error_handler(reason)
            except Exception:  # noqa: BLE001
                logger.exception("处理通道错误失败")

    # ---------- 生命周期 ----------
    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def send(self, payload: bytes) -> bool:
        """返回是否成功写入通道。"""

    # ---------- 状态 ----------
    @property
    @abc.abstractmethod
    def connected(self) -> bool: ...

    @property
    def port_name(self) -> str:
        return ""

    @property
    def simulated(self) -> bool:
        return False

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "connected": self.connected,
            "port": self.port_name,
            "simulated": self.simulated,
        }
