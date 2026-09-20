"""报警控制。

安全约束：
- 模拟模式下绝不向真实串口下发报警命令（通道非模拟时直接拒绝并记录 blocked）；
- 真实设备测试必须带二次确认标记；
- 临时静音只关闭蜂鸣器，红灯与报警状态保留；
- 烟感等硬报警由规则引擎判定，Qwen 无权取消。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from ..config.runtime import RuntimeConfig
from ..database.repositories import Repository
from ..devices.manager import DeviceManager
from ..schemas.common import AlertState
from ..schemas.enums import AlertMode, BuzzerMode, LedColor, SafetyState

logger = logging.getLogger(__name__)

ALARM_NODE = "alarm_01"

SAFETY_TO_OUTPUT: dict[SafetyState, tuple[LedColor, BuzzerMode]] = {
    SafetyState.NORMAL: (LedColor.GREEN, BuzzerMode.OFF),
    SafetyState.NOTICE: (LedColor.BLUE, BuzzerMode.OFF),
    SafetyState.WARNING: (LedColor.YELLOW, BuzzerMode.OFF),
    SafetyState.ALARM: (LedColor.RED, BuzzerMode.CONTINUOUS),
}


class AlertController:
    def __init__(
        self,
        devices: DeviceManager,
        runtime: RuntimeConfig,
        repo: Repository,
        *,
        broadcast: Callable[[str, Any], None] | None = None,
    ) -> None:
        self.devices = devices
        self.runtime = runtime
        self.repo = repo
        self._broadcast = broadcast
        self.state = AlertState(simulated=runtime.simulation_mode)
        self._lock = asyncio.Lock()
        self._last_output: tuple[LedColor, BuzzerMode] | None = None
        self._mode = AlertMode.AUTO
        self._muted_until: float | None = None
        self._test_task: asyncio.Task | None = None

    # ---------- 内部 ----------
    def _can_send_real(self) -> bool:
        """模拟模式下只允许模拟通道，杜绝误触发真实蜂鸣器。"""
        if not self.runtime.simulation_mode:
            return True
        return bool(getattr(self.devices.adapter, "simulated", False))

    def _publish(self) -> None:
        self.state.simulated = bool(getattr(self.devices.adapter, "simulated", True))
        self.state.mode = self._mode.value
        self.state.muted_until = self._muted_until
        if self._broadcast:
            self._broadcast("alert_state", self.state.model_dump(mode="json"))

    async def _dispatch(
        self,
        color: LedColor,
        buzzer: BuzzerMode,
        *,
        event_id: str | None = None,
        reason: str = "",
        force: bool = False,
    ) -> dict[str, Any]:
        if not force and self._last_output == (color, buzzer):
            return {"ok": True, "status": "unchanged"}

        payload = {
            "color": color.value,
            "buzzer": buzzer != BuzzerMode.OFF,
            "mode": buzzer.value,
            "reason": reason[:64],
        }
        if not self._can_send_real():
            self.repo.insert_alert_action(
                event_id=event_id,
                request_id=None,
                target=ALARM_NODE,
                action="set_alarm",
                payload=payload,
                status="blocked",
                simulated=True,
            )
            logger.warning("模拟模式下拦截了真实报警指令：%s", payload)
            return {"ok": False, "status": "blocked", "reason": "模拟模式禁止操作真实报警设备"}

        result = await self.devices.send_command(ALARM_NODE, "set_alarm", payload)
        simulated = bool(getattr(self.devices.adapter, "simulated", True))
        self.repo.insert_alert_action(
            event_id=event_id,
            request_id=result.get("request_id"),
            target=ALARM_NODE,
            action="set_alarm",
            payload=payload,
            status=("ack" if result.get("ok") else result.get("status", "failed")),
            simulated=simulated,
            retries=int(result.get("retries", 0)),
            acked_at=time.time() if result.get("ok") else None,
        )
        self._last_output = (color, buzzer)
        self.state.led_color = color
        self.state.buzzer = buzzer
        self.state.last_command_at = time.time()
        if result.get("ok"):
            self.state.last_ack_at = time.time()
        else:
            logger.warning("报警指令未确认: %s", result)
        # OLED 同步显示当前状态（失败不影响报警本身）
        await self.devices.send_command(
            ALARM_NODE,
            "set_display",
            {"line1": self.state.safety_state.value, "line2": reason[:16]},
            wait_ack=False,
        )
        return result

    # ---------- 自动模式 ----------
    async def apply_safety_state(
        self, safety: SafetyState, reason: str, event_id: str | None = None
    ) -> dict[str, Any]:
        async with self._lock:
            self.state.safety_state = safety
            self.state.reason = reason
            self.state.active = safety.severity >= SafetyState.WARNING.severity
            color, buzzer = SAFETY_TO_OUTPUT[safety]
            if self._muted_until and time.time() < self._muted_until:
                buzzer = BuzzerMode.OFF  # 静音只影响蜂鸣器
            elif self._muted_until:
                self._muted_until = None
            if self._mode is AlertMode.MANUAL_TEST:
                self._publish()
                return {"ok": True, "status": "manual_test_mode"}
            result = await self._dispatch(color, buzzer, event_id=event_id, reason=reason)
            self._publish()
            return result

    # ---------- 手动操作 ----------
    async def mute(self, seconds: float, operator: str = "local") -> dict[str, Any]:
        async with self._lock:
            self._muted_until = time.time() + max(1.0, seconds)
            self.repo.insert_operator_action(None, "alert_mute", operator, f"{seconds:.0f}s")
            result = await self._dispatch(
                self.state.led_color, BuzzerMode.OFF, reason="临时静音", force=True
            )
            self._publish()
            return {"ok": True, "muted_until": self._muted_until, "detail": result}

    async def unmute(self, operator: str = "local") -> dict[str, Any]:
        async with self._lock:
            self._muted_until = None
            self.repo.insert_operator_action(None, "alert_unmute", operator, None)
            color, buzzer = SAFETY_TO_OUTPUT[self.state.safety_state]
            result = await self._dispatch(color, buzzer, reason=self.state.reason, force=True)
            self._publish()
            return {"ok": True, "detail": result}

    async def acknowledge(self, operator: str = "local", note: str | None = None) -> dict[str, Any]:
        self.repo.insert_operator_action(None, "alert_acknowledge", operator, note)
        self._publish()
        return {"ok": True, "message": "已确认当前报警"}

    async def clear(self, operator: str = "local") -> dict[str, Any]:
        """人工解除：回到正常输出（若风险仍存在，规则引擎会再次拉起）。"""
        async with self._lock:
            self.repo.insert_operator_action(None, "alert_clear", operator, None)
            self.state.safety_state = SafetyState.NORMAL
            self.state.active = False
            self.state.reason = "人工解除"
            result = await self._dispatch(
                LedColor.GREEN, BuzzerMode.OFF, reason="人工解除", force=True
            )
            self._publish()
            return {"ok": True, "detail": result}

    async def test(
        self,
        *,
        target: str = "led",
        color: str = "red",
        buzzer_mode: str = "pulse",
        duration_seconds: float = 3.0,
        confirm: bool = False,
    ) -> dict[str, Any]:
        """手动测试。真实模式必须 confirm=True（前端二次确认）。"""
        real = not self.runtime.simulation_mode
        if real and not confirm:
            return {"ok": False, "message": "真实设备测试需要二次确认"}
        if self.runtime.simulation_mode and not self._can_send_real():
            self.repo.insert_alert_action(
                event_id=None,
                request_id=None,
                target=ALARM_NODE,
                action=f"test_{target}",
                payload={"color": color, "buzzer_mode": buzzer_mode},
                status="blocked",
                simulated=True,
            )
            return {"ok": False, "message": "模拟模式禁止操作真实报警设备"}

        async with self._lock:
            self._mode = AlertMode.MANUAL_TEST
            led = LedColor(color) if target in {"led", "both"} else self.state.led_color
            buzzer = (
                BuzzerMode(buzzer_mode) if target in {"buzzer", "both"} else BuzzerMode.OFF
            )
            result = await self._dispatch(led, buzzer, reason="手动测试", force=True)
            self.repo.insert_operator_action(
                None, "alert_test", "local", f"{target}/{color}/{buzzer_mode}"
            )
            self._publish()

        async def _restore() -> None:
            await asyncio.sleep(max(0.5, duration_seconds))
            async with self._lock:
                self._mode = AlertMode.AUTO
            await self.apply_safety_state(self.state.safety_state, self.state.reason)

        if self._test_task and not self._test_task.done():
            self._test_task.cancel()
        self._test_task = asyncio.create_task(_restore(), name="alert-test-restore")
        return {"ok": True, "detail": result, "restore_after": duration_seconds}

    # ---------- 查询 ----------
    def snapshot(self) -> AlertState:
        self.state.simulated = bool(getattr(self.devices.adapter, "simulated", True))
        self.state.mode = self._mode.value
        self.state.muted_until = self._muted_until
        return self.state

    def recent_actions(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.repo.list_alert_actions(limit)
