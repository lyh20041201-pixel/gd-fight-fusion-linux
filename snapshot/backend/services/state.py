"""应用状态中枢：组装所有模块并驱动主循环。

主循环（默认 2 Hz）：
    摄像头人数融合 -> 组装教室快照 -> 本地规则引擎 -> 报警执行 -> 事件建档 -> WebSocket 推送
Qwen 复核在独立队列中异步执行，失败不影响任何本地链路。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..ai.qwen_reviewer import QwenResult, QwenReviewer
from ..ai.queue import QwenQueue
from ..audio.capture import MicrophoneCapture
from ..alerts.controller import AlertController
from ..cameras.manager import CameraConfig, CameraManager
from ..config.defaults import DEFAULT_CAMERA_REGIONS, DEFAULT_RULES
from ..config.runtime import RuntimeConfig
from ..config.settings import Settings
from ..config.vision import load_vision_config
from ..database.db import Database, DatabaseError
from ..database.repositories import Repository
from ..devices.manager import DEFAULT_NODES, SENSOR_META, DeviceManager
from ..devices.serial_gateway import SerialGateway
from ..devices.simulator import SimulatedGateway
from ..fusion.occupancy import OccupancyFusion
from ..fusion.regions import default_roi_for_index
from ..rules.engine import CameraSnapshot, ClassroomSnapshot, RuleEngine, RuleOutcome
from ..schemas.common import (
    AlertState,
    ModuleStatus,
    OccupancyUpdate,
    SystemHealth,
    SystemStatus,
)
from ..schemas.enums import HealthState, ModuleState, ModuleType, SafetyState
from ..vision.detector import create_detector
from .events import RiskEventService, purge_event_images
from .visual_events import VisualEventService, analyze_cameras
from ..vision.video_events import VideoEventDetector
from .live_actions import LiveActionService
from .ws_hub import WsHub

logger = logging.getLogger(__name__)

CONTROL_INTERVAL = 0.5
BROADCAST_INTERVAL = 1.0
PERSIST_INTERVAL = 15.0
MAINTENANCE_INTERVAL = 3600.0


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.started_at = time.time()

        # 数据库
        self.db = Database(settings.db_file)
        self.repo = Repository(self.db)
        self.runtime = RuntimeConfig(settings, self.repo)

        # 实时推送
        self.ws = WsHub()

        # 视觉与融合
        self.fusion = OccupancyFusion(
            median_window_seconds=3.0,
            spike_window_seconds=10.0,
        )
        self.detector = None  # type: ignore[assignment]
        self.cameras: CameraManager | None = None
        self.vision_config = None  # 启动时从 config/vision.yaml 载入

        # 设备
        self.devices: DeviceManager | None = None

        # 规则 / 报警 / 事件 / Qwen
        self.rules: RuleEngine | None = None
        self.alerts: AlertController | None = None
        self.events: RiskEventService | None = None
        self.qwen = QwenReviewer(settings, self.runtime)
        self.audio = MicrophoneCapture(settings, self.runtime)
        self._audio_config_lock = asyncio.Lock()
        self.qwen_queue: QwenQueue | None = None
        self.visual_events = None
        self.live_actions = None

        self._tasks: list[asyncio.Task] = []
        self._last_broadcast = 0.0
        self._last_persist = 0.0
        self._last_health_broadcast = 0.0
        self._last_alert_state: str | None = None
        self._degraded: list[str] = []
        self._ready = False

    # ================= 启动 =================
    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self.ws.bind_loop(loop)

        self.db.init_schema()
        self.runtime.load()
        self._seed_database()

        simulation = self.runtime.simulation_mode
        self.audio.startup_simulation = simulation
        logger.info("运行模式: %s", "模拟模式" if simulation else "真实模式")

        # ---- 视觉 ----
        # 算法参数优先取 config/vision.yaml，没有该文件时回落到 .env 与代码默认值
        self.vision_config = load_vision_config(self.settings.base_dir)
        det_cfg = self.vision_config.detector
        model_path = (
            self.settings.resolve_path(det_cfg.model_path)
            if det_cfg.model_path
            else self.settings.model_file
        )
        detector, degraded = create_detector(
            simulation_mode=simulation,
            model_path=model_path,
            confidence=det_cfg.conf_threshold,
            device=det_cfg.device,
            vision_enabled=self.settings.vision_enabled,
            iou_threshold=det_cfg.iou_threshold,
            person_class=det_cfg.person_class,
            max_det=det_cfg.max_det,
            imgsz=det_cfg.imgsz,
        )
        self.detector = detector
        self.cameras = CameraManager(
            self.settings,
            self.runtime,
            detector,
            self.fusion,
            detector_degraded=degraded,
            vision_config=self.vision_config,
        )
        self.cameras.build(self._camera_configs())
        self.cameras.start()
        self.audio.start()

        # ---- 设备 ----
        adapter = (
            SimulatedGateway()
            if simulation
            else SerialGateway(
                port=str(self.runtime.get("serial_port", "") or ""),
                baudrate=self.settings.serial_baudrate,
                auto_discover=self.settings.serial_auto_discover,
                reconnect_seconds=self.settings.serial_reconnect_seconds,
            )
        )
        self.devices = DeviceManager(
            adapter,
            self.runtime,
            ack_timeout=self.settings.command_ack_timeout,
            max_retries=self.settings.command_max_retries,
            heartbeat_timeout=self.settings.heartbeat_timeout_seconds,
        )
        self.devices.on_critical_change = self._on_critical_sensor
        await self.devices.start()

        # ---- 规则 / 报警 ----
        self.rules = RuleEngine(self.runtime, self._load_rules())
        self.alerts = AlertController(
            self.devices, self.runtime, self.repo, broadcast=self.ws.broadcast
        )

        # ---- Qwen ----
        self.qwen_queue = QwenQueue(
            self.qwen, self._on_qwen_result, maxsize=self.settings.qwen_max_queue
        )
        await self.qwen_queue.start()

        # ---- 事件服务 ----
        self.events = RiskEventService(
            self.repo,
            self.cameras,
            self.runtime,
            self.settings.events_dir,
            broadcast=self.ws.broadcast,
            qwen_queue=self.qwen_queue,
            audio=self.audio,
        )

        self._tasks = [
            asyncio.create_task(self._control_loop(), name="control-loop"),
            asyncio.create_task(self._persist_loop(), name="persist-loop"),
            asyncio.create_task(self._maintenance_loop(), name="maintenance-loop"),
        ]
        self.visual_events = VisualEventService(self.repo, self.settings.events_dir,
            self.qwen_queue, self.ws.broadcast, self._visual_alarm, audio=self.audio, cameras=self.cameras)
        await self.visual_events.recover()
        self.live_actions = LiveActionService(self.settings, self.cameras, self.visual_events,
                                             self.ws.broadcast, simulation)
        await self.live_actions.start()
        detectors = []
        for path in ([] if simulation else self.settings.video_event_checkpoints):
            try:
                detectors.append(await asyncio.to_thread(VideoEventDetector, path, self.settings.video_event_device))
            except Exception as exc:
                logger.exception("视频事件权重加载失败 %s", path)
                self.repo.insert_log("ERROR", "video_events", "视频事件模型未启用", {"path":path,"error":str(exc)})
        if detectors:
            self._tasks.append(asyncio.create_task(analyze_cameras(self.cameras,detectors,self.visual_events),name="video-events"))
        self._ready = True
        self.repo.insert_log(
            "INFO", "system", "系统启动", {"simulation": simulation, "detector": detector.backend}
        )

    async def stop(self) -> None:
        self._ready = False
        if self.live_actions:
            await self.live_actions.stop()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks.clear()

        if self.events:
            await self.events.close()
        if self.qwen_queue:
            await self.qwen_queue.stop()
        await self.qwen.close()
        await asyncio.to_thread(self.audio.stop)
        if self.devices:
            await self.devices.stop()
        if self.cameras:
            await asyncio.to_thread(self.cameras.stop)
        try:
            self.repo.insert_log("INFO", "system", "系统关闭", {})
        except DatabaseError:
            pass
        self.db.close()
        logger.info("所有资源已释放")

    # ================= 初始化数据 =================
    def _seed_database(self) -> None:
        for node in DEFAULT_NODES:
            self.repo.upsert_device(
                node.node_id,
                node.name,
                node.module_type.value,
                capabilities=node.capabilities,
            )
            for kind in node.sensors:
                label, unit = SENSOR_META.get(kind, (kind, ""))
                self.repo.upsert_sensor(
                    f"{node.node_id}.{kind}", node.node_id, kind, label, unit
                )
        for rule in DEFAULT_RULES:
            self.repo.upsert_rule(rule)

        existing = {c["camera_id"] for c in self.repo.list_cameras()}
        count = (
            max(1, self.settings.simulated_camera_count)
            if self.runtime.simulation_mode
            else len(self.runtime.get("camera_device_indices") or [])
        )
        for i in range(count):
            camera_id = f"CAM-{i + 1:02d}"
            if camera_id in existing:
                # 模拟模式下的摄像头由系统生成，必须是启用的。
                # 否则用户在真实模式下把设备索引清空（cameras.enabled 置 0）之后，
                # 再切回模拟模式会一路画面都没有 —— 演示时很容易踩到这个坑。
                if self.runtime.simulation_mode:
                    self.repo.update_camera_config(camera_id, enabled=True)
                continue
            region, is_primary = DEFAULT_CAMERA_REGIONS[i % len(DEFAULT_CAMERA_REGIONS)]
            self.repo.upsert_camera(
                camera_id,
                f"摄像头 {i + 1}",
                region,
                f"sim:{i}" if self.runtime.simulation_mode else f"usb:{i}",
                default_roi_for_index(i),
                is_primary_overlap=is_primary,
            )

    def _camera_configs(self) -> list[CameraConfig]:
        configs: list[CameraConfig] = []
        for row in self.repo.list_cameras():
            configs.append(
                CameraConfig(
                    camera_id=row["camera_id"],
                    name=row["name"],
                    region=row.get("region", ""),
                    is_primary_overlap=bool(row.get("is_primary_overlap", 1)),
                    roi=row.get("roi") or [],
                    enabled=bool(row.get("enabled", 1)),
                )
            )
        return configs

    def _load_rules(self) -> list[dict[str, Any]]:
        stored = {r["rule_id"]: r for r in self.repo.list_rules()}
        rules: list[dict[str, Any]] = []
        for default in DEFAULT_RULES:
            rule = dict(default)
            row = stored.get(rule["rule_id"])
            if row:
                rule["enabled"] = bool(row.get("enabled", True))
                rule["risk_level"] = row.get("risk_level", rule["risk_level"])
                params = dict(rule.get("params", {}))
                params.update(row.get("params") or {})
                rule["params"] = params
            rules.append(rule)
        return rules

    # ================= 主循环 =================
    async def _control_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(CONTROL_INTERVAL)
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - 主循环必须永不退出
                logger.exception("控制循环异常")
                await asyncio.sleep(1.0)

    async def _tick(self) -> None:
        if not (self.cameras and self.devices and self.rules and self.alerts and self.events):
            return
        now = time.time()
        self.cameras.refresh_fusion()
        snapshot = self._build_snapshot(now)

        # 模拟环境数据与人数联动，使模拟数据自洽
        self.devices.set_people_hint(snapshot.occupancy_total)

        outcomes = self.rules.evaluate(snapshot)
        await self._handle_outcomes(outcomes, snapshot)

        if now - self._last_broadcast >= BROADCAST_INTERVAL:
            self._last_broadcast = now
            self._broadcast_realtime(snapshot)
        if now - self._last_health_broadcast >= 2.0:
            self._last_health_broadcast = now
            self.ws.broadcast("system_health", self.health().model_dump(mode="json"))
            self.ws.broadcast(
                "module_status", [m.model_dump(mode="json") for m in self.modules()]
            )

    def _build_snapshot(self, now: float) -> ClassroomSnapshot:
        assert self.cameras and self.devices
        sensors = self.devices.latest_values()
        sensor_online = {
            kind: self.devices.sensor_online(kind) for kind in SENSOR_META
        }
        camera_snaps: list[CameraSnapshot] = []
        offline_cameras: list[str] = []
        for worker in self.cameras.workers.values():
            snap = worker.snapshot()
            if not snap.online and snap.last_frame_at:
                offline_cameras.append(snap.camera_id)
            camera_snaps.append(
                CameraSnapshot(
                    camera_id=snap.camera_id,
                    online=snap.online,
                    obstructed=snap.obstructed,
                    person_count=snap.person_count,
                    region=snap.region,
                    track_labels=[t.label for t in snap.tracks],
                    low_confidence=any(t.confidence < 0.45 for t in snap.tracks),
                )
            )
        # 只有"曾经在线过"的设备才算掉线；从未连接过的硬件属于未接入，
        # 界面上照常显示离线/无数据，但不会反复生成"关键设备离线"事件
        offline_devices = [
            d.device_id
            for d in self.devices.devices()
            if not d.online and (d.last_seen or d.last_heartbeat)
        ]
        conflicts = self.fusion.region_conflicts(
            int(self.runtime.threshold("camera_conflict_delta", 3))
        )
        return ClassroomSnapshot(
            timestamp=now,
            sensors=sensors,
            sensor_online=sensor_online,
            occupancy_total=self.fusion.smoothed_total(),
            occupancy_raw=self.fusion.raw_total(),
            occupancy_spike=self.fusion.spike_delta(),
            cameras=camera_snaps,
            conflicts=conflicts,
            offline_devices=offline_devices,
            offline_cameras=offline_cameras,
        )

    async def _handle_outcomes(
        self, outcomes: list[RuleOutcome], snapshot: ClassroomSnapshot
    ) -> None:
        assert self.rules and self.alerts and self.events and self.cameras
        triggers = [o for o in outcomes if o.kind == "trigger"]
        clears = [o for o in outcomes if o.kind == "clear"]

        safety, reason = self.rules.safety_state()
        safety, reason = self._with_visual_alarm(safety, reason)

        # 1) 先执行本地声光报警（不等待图片保存与 Qwen）
        alert_result: dict[str, Any] = {}
        if triggers or clears or safety.value != self.alerts.state.safety_state.value:
            alert_result = await self.alerts.apply_safety_state(safety, reason)

        # 2) 再建立事件档案
        for outcome in triggers:
            # 先在画面上标记风险，保证随后抓取的关键帧带有风险框与风险类型
            self._mark_risk(outcome)
            context = self._event_context(snapshot, safety, alert_result, outcome)
            event = await self.events.create_from_outcome(outcome, context)
            self.rules.mark_event(outcome.rule_id, event.event_id)

        for outcome in clears:
            if outcome.camera_id:
                worker = self.cameras.get(outcome.camera_id)
                if worker:
                    worker.clear_risk()
            logger.info("规则解除: %s (%s)", outcome.rule_id, outcome.description)

        if not self.rules.active_rules():
            for worker in self.cameras.workers.values():
                worker.clear_risk()

    def _mark_risk(self, outcome: RuleOutcome) -> None:
        """把风险标记同步到摄像头标注流水线。

        注意：只标记"已有的" Track，绝不虚构检测框（框只能来自 YOLO / ByteTrack）。
        """
        if self.cameras is None:
            return
        if outcome.camera_id:
            worker = self.cameras.get(outcome.camera_id)
            if worker is None:
                return
            track_ids = [
                t.track_id
                for t in worker.tracks
                if not outcome.track_ids
                or f"{outcome.camera_id}-{t.label}" in outcome.track_ids
            ]
            worker.set_risk(track_ids, outcome.title)
            return
        # 传感器/融合类事件没有具体目标，只在所有在线画面上标注风险类型
        for worker in self.cameras.workers.values():
            if worker.online:
                worker.set_risk([], outcome.title)

    def _event_context(
        self,
        snapshot: ClassroomSnapshot,
        safety: SafetyState,
        alert_result: dict[str, Any],
        outcome: RuleOutcome,
    ) -> dict[str, Any]:
        assert self.cameras
        track_summary: dict[str, Any] = {}
        if outcome.camera_id:
            worker = self.cameras.get(outcome.camera_id)
            if worker:
                track_summary = worker.track_summary()
        else:
            track_summary = {
                camera_id: worker.track_summary()
                for camera_id, worker in self.cameras.workers.items()
                if worker.online
            }
        return {
            "safety_state": safety.value,
            "sensors": snapshot.sensors,
            "occupancy": snapshot.occupancy_total,
            "per_camera": self.fusion.per_camera(),
            "track_summary": track_summary,
            "camera_summary": {
                c.camera_id: {
                    "online": c.online,
                    "count": c.person_count,
                    "region": c.region,
                    "obstructed": c.obstructed,
                }
                for c in snapshot.cameras
            },
            "alert_result": alert_result,
        }

    def _broadcast_realtime(self, snapshot: ClassroomSnapshot) -> None:
        assert self.cameras
        self.ws.broadcast(
            "sensor_update",
            {
                "values": snapshot.sensors,
                "online": snapshot.sensor_online,
                "timestamp": snapshot.timestamp,
            },
        )
        self.ws.broadcast(
            "occupancy_update",
            OccupancyUpdate(
                total=snapshot.occupancy_total,
                per_camera=self.fusion.per_camera(),
                timestamp=snapshot.timestamp,
                trend=self.fusion.trend(),
            ).model_dump(mode="json"),
        )
        self.ws.broadcast(
            "camera_status",
            [c.model_dump(mode="json") for c in self.cameras.snapshots()],
        )

    # ================= 传感器紧急事件 =================
    def _on_critical_sensor(self, node_id: str, value: float) -> None:
        """烟感等硬报警：立即触发一次规则评估，不等下一个循环周期。"""
        logger.warning("收到紧急传感器变化 %s=%s，立即评估规则", node_id, value)
        asyncio.create_task(self._immediate_tick())

    async def _immediate_tick(self) -> None:
        try:
            await self._tick()
        except Exception:  # noqa: BLE001
            logger.exception("紧急评估失败")

    # ================= 持久化 =================
    async def _persist_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(PERSIST_INTERVAL)
                await asyncio.to_thread(self._persist_samples)
            except asyncio.CancelledError:
                raise
            except DatabaseError as exc:
                logger.error("采样落库失败: %s", exc)
            except Exception:  # noqa: BLE001
                logger.exception("持久化循环异常")

    def _persist_samples(self) -> None:
        if not (self.devices and self.cameras):
            return
        now = time.time()
        rows: list[tuple[str, str, str, float, str, float]] = []
        for node in self.devices.node_stats():
            for kind, value in node.values.items():
                label, unit = SENSOR_META.get(kind, (kind, ""))
                rows.append(
                    (f"{node.node_id}.{kind}", node.node_id, kind, float(value), unit, now)
                )
        self.repo.insert_readings(rows)
        self.repo.insert_occupancy(None, self.fusion.smoothed_total(), now)
        for camera_id, count in self.fusion.per_camera().items():
            self.repo.insert_occupancy(camera_id, count, now)

    async def _maintenance_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(MAINTENANCE_INTERVAL)
                retention_days = self.runtime.data_retention_days
                removed, images = await asyncio.to_thread(
                    self.repo.purge_old_data, retention_days
                )
                deleted = await asyncio.to_thread(
                    purge_event_images, self.settings.events_dir, images, retention_days
                )
                logger.info("历史数据清理完成: %s，事件图片 %d 个", removed, deleted)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("维护任务异常")

    # ================= Qwen 回调 =================
    async def _on_qwen_result(self, event_id: str, result: QwenResult) -> None:
        if self.visual_events and await self.visual_events.on_result(event_id, result):
            return
        if self.events is None:
            return
        try:
            await self.events.store_qwen_result(event_id, result)
        except Exception:  # noqa: BLE001 - Qwen 失败不得影响其它模块
            logger.exception("写入 Qwen 结果失败: %s", event_id)

    def _with_visual_alarm(self, safety, reason):
        row = self.repo.db.query_one("SELECT title FROM risk_events WHERE status='PENDING' "
            "AND json_extract(rule_basis,'$.visual.state')='confirmed' "
            "AND json_extract(rule_basis,'$.visual.alarm_held')=1 LIMIT 1")
        if row and safety.severity < SafetyState.ALARM.severity:
            return SafetyState.ALARM, row["title"] + "（复核确认）"
        return safety, reason

    async def _visual_alarm(self, event):
        safety, reason = self.rules.safety_state()
        safety, reason = self._with_visual_alarm(safety, reason)
        return await self.alerts.apply_safety_state(safety, reason, event_id=event.event_id)

    # ================= 对外状态 =================
    def modules(self) -> list[ModuleStatus]:
        modules: list[ModuleStatus] = []
        if self.live_actions:
            action = self.live_actions
            modules.append(ModuleStatus(module_id='service.actions', name='跌倒与打架动作识别',
                module_type=ModuleType.SERVICE,
                state=ModuleState.ONLINE if action.state == 'ready' else ModuleState.FAULT if action.state == 'error' else ModuleState.OFFLINE,
                value_text='本地识别运行中' if action.state == 'ready' else '未就绪',
                fault_reason=action.reason, detail=action.status()))
        audio = self.audio.status()
        modules.append(ModuleStatus(module_id='service.audio', name='麦克风与音频缓存',
            module_type=ModuleType.SERVICE,
            state=ModuleState.ONLINE if audio['online'] else ModuleState.OFFLINE,
            value_text=audio['device_name'] if audio['online'] else '未采集',
            detail=audio, last_update=audio['last_sample_at'], fault_reason=audio['error']))
        if self.devices:
            modules += self.devices.module_statuses()
        if self.cameras:
            modules += self.cameras.module_statuses()
            modules += self.cameras.vision_module_statuses()
        modules += self._service_modules()
        return modules

    def _service_modules(self) -> list[ModuleStatus]:
        now = time.time()
        db_ok = self.db.healthy
        qwen_status = self.qwen.status()
        qwen_state = ModuleState.ONLINE
        qwen_reason = None
        if not qwen_status["enabled"]:
            qwen_state, qwen_reason = ModuleState.OFFLINE, "已在系统设置中停用"
        elif not qwen_status["configured"]:
            qwen_state, qwen_reason = ModuleState.OFFLINE, "未配置 API 密钥（环境变量）"
        elif qwen_status["last_error"]:
            qwen_state, qwen_reason = ModuleState.WARNING, str(qwen_status["last_error"])

        conflicts = self.fusion.region_conflicts(
            int(self.runtime.threshold("camera_conflict_delta", 3))
        )
        rules_active = self.rules.active_rules() if self.rules else []

        return [
            ModuleStatus(
                module_id="service.fusion",
                name="多摄像头区域融合",
                module_type=ModuleType.SERVICE,
                state=ModuleState.WARNING if conflicts else ModuleState.ONLINE,
                value_text=f"全局 {self.fusion.smoothed_total()} 人",
                detail=self.fusion.snapshot(),
                last_update=now,
                fault_reason="存在区域人数矛盾" if conflicts else None,
            ),
            ModuleStatus(
                module_id="service.rules",
                name="本地规则引擎",
                module_type=ModuleType.SERVICE,
                state=ModuleState.ALARM if rules_active else ModuleState.ONLINE,
                value_text=(f"{len(rules_active)} 条规则命中" if rules_active else "全部正常"),
                detail=self.rules.snapshot() if self.rules else {},
                last_update=now,
            ),
            ModuleStatus(
                module_id="service.qwen",
                name="Qwen 多模态复核",
                module_type=ModuleType.SERVICE,
                state=qwen_state,
                value_text=(
                    f"{qwen_status['model']} · 调用 {qwen_status['calls']} 次"
                    if qwen_status["configured"]
                    else "未配置"
                ),
                detail={
                    **{k: v for k, v in qwen_status.items() if k != "last_error"},
                    "queue": self.qwen_queue.stats() if self.qwen_queue else {},
                },
                last_update=now,
                fault_reason=qwen_reason,
            ),
            ModuleStatus(
                module_id="service.database",
                name="SQLite 数据存储",
                module_type=ModuleType.SERVICE,
                state=ModuleState.ONLINE if db_ok else ModuleState.FAULT,
                value_text=str(self.settings.db_file.name),
                detail={"path": str(self.settings.db_file), "wal": True},
                last_update=now,
                fault_reason=None if db_ok else (self.db.last_error or "数据库异常"),
            ),
            ModuleStatus(
                module_id="service.websocket",
                name="WebSocket 实时推送",
                module_type=ModuleType.SERVICE,
                state=ModuleState.ONLINE,
                value_text=f"{self.ws.client_count} 个客户端",
                detail={"clients": self.ws.client_count, "messages": self.ws.sent},
                last_update=now,
            ),
        ]

    def health(self) -> SystemHealth:
        modules = self.modules()
        qwen_optional = not (self.qwen.configured and self.qwen.enabled)
        degraded = [
            m.module_id
            for m in modules
            if m.state in {ModuleState.OFFLINE, ModuleState.FAULT, ModuleState.WARNING}
            and m.module_type != ModuleType.SENSOR
            # Qwen 未配置/已停用属于可选能力，不算系统降级（系统必须能完整运行）
            and not (m.module_id == "service.qwen" and qwen_optional)
            and not (m.module_id == 'service.audio' and (self.runtime.simulation_mode or not self.runtime.get('audio_enabled')))
        ]
        faults = [m for m in modules if m.state is ModuleState.FAULT]
        offline = [
            m
            for m in modules
            if m.state is ModuleState.OFFLINE
            and not (m.module_id == "service.qwen" and qwen_optional)
            and not (m.module_id == 'service.audio' and (self.runtime.simulation_mode or not self.runtime.get('audio_enabled')))
        ]
        if faults:
            health_state = HealthState.FAULT
        elif degraded or offline:
            health_state = HealthState.DEGRADED
        else:
            health_state = HealthState.HEALTHY

        safety = SafetyState.NORMAL
        if self.rules:
            safety, _ = self.rules.safety_state()
        qwen_status = self.qwen.status()
        return SystemHealth(
            health_state=health_state,
            safety_state=safety,
            simulation_mode=self.runtime.simulation_mode,
            qwen_configured=bool(qwen_status["configured"]),
            qwen_available=bool(qwen_status["available"]),
            qwen_enabled=bool(qwen_status["enabled"]),
            database_ok=self.db.healthy,
            serial_connected=bool(self.devices and self.devices.gateway_connected()),
            serial_port=(self.devices.adapter.port_name if self.devices else ""),
            ws_clients=self.ws.client_count,
            vision_backend=(self.detector.backend if self.detector else "none"),
            uptime_seconds=round(time.time() - self.started_at, 1),
            degraded_modules=degraded,
            server_time=time.time(),
        )

    def alert_state(self) -> AlertState:
        if self.alerts:
            return self.alerts.snapshot()
        return AlertState()

    def occupancy(self) -> OccupancyUpdate:
        return OccupancyUpdate(
            total=self.fusion.smoothed_total(),
            per_camera=self.fusion.per_camera(),
            timestamp=time.time(),
            trend=self.fusion.trend(),
        )

    def system_status(self) -> SystemStatus:
        sensors = self.devices.latest_values() if self.devices else {}
        return SystemStatus(
            health=self.health(),
            alert=self.alert_state(),
            occupancy=self.occupancy(),
            environment={
                "temperature": sensors.get("temperature"),
                "humidity": sensors.get("humidity"),
                "co2": sensors.get("co2"),
                "light": sensors.get("light"),
                "noise": sensors.get("noise"),
                "smoke": sensors.get("smoke"),
            },
            active_events=self.repo.count_events(status="PENDING"),
            pending_events=self.repo.count_events(status="PENDING"),
        )

    def ws_snapshot(self) -> dict[str, Any]:
        """WebSocket 连接建立时的一次性全量快照。"""
        _total, events = self.repo.list_events(limit=20)
        return {
            "status": self.system_status().model_dump(mode="json"),
            "modules": [m.model_dump(mode="json") for m in self.modules()],
            "cameras": [
                c.model_dump(mode="json") for c in (self.cameras.snapshots() if self.cameras else [])
            ],
            "events": [e.model_dump(mode="json") for e in events],
        }

    # ================= 摄像头接入变更 =================
    async def apply_camera_selection(self, indices: list[int]) -> dict[str, Any]:
        """更换接入的摄像头设备（运行时生效，无需重启）。"""
        if self.cameras is None:
            return {"ok": False, "message": "摄像头服务未就绪"}
        if self.runtime.simulation_mode:
            return {
                "ok": False,
                "message": "模拟模式下摄像头由系统生成，切换到真实模式后才能选择设备",
            }

        clean = [int(i) for i in indices if int(i) >= 0]
        await asyncio.to_thread(self._sync_camera_rows, clean)
        configs = await asyncio.to_thread(self._camera_configs)
        await asyncio.to_thread(self.cameras.rebuild, configs)
        self.ws.broadcast(
            "camera_status", [c.model_dump(mode="json") for c in self.cameras.snapshots()]
        )
        self.repo.insert_log("INFO", "cameras", "更新接入的摄像头设备", {"indices": clean})
        return {"ok": True, "cameras": len(self.cameras.workers), "indices": clean}

    def _sync_camera_rows(self, indices: list[int]) -> None:
        """按选中的设备索引维护 cameras 表：多退少补，保留已有的名称/区域/ROI。"""
        existing = {row["camera_id"]: row for row in self.repo.list_cameras()}
        for position, device_index in enumerate(indices):
            camera_id = f"CAM-{position + 1:02d}"
            region, is_primary = DEFAULT_CAMERA_REGIONS[position % len(DEFAULT_CAMERA_REGIONS)]
            row = existing.get(camera_id)
            self.repo.upsert_camera(
                camera_id,
                row["name"] if row else f"摄像头 {position + 1}",
                row["region"] if row else region,
                f"usb:{device_index}",
                (row.get("roi") if row else None) or default_roi_for_index(position),
                is_primary_overlap=(
                    bool(row["is_primary_overlap"]) if row else is_primary
                ),
            )
            self.repo.update_camera_config(camera_id, enabled=True)
        # 多出来的旧摄像头置为停用，不再构建
        for position in range(len(indices), len(existing)):
            self.repo.update_camera_config(f"CAM-{position + 1:02d}", enabled=False)

    # ================= 模拟控制 =================
    async def simulate(self, scenario: str, node_id: str | None = None,
                       camera_id: str | None = None) -> dict[str, Any]:
        if not self.runtime.simulation_mode:
            return {"ok": False, "message": "真实模式下不支持场景注入"}
        detail: dict[str, Any] = {}
        if self.devices:
            detail["devices"] = self.devices.inject_scenario(scenario, node_id)
        if self.cameras:
            detail["cameras"] = self.cameras.inject(scenario, camera_id)
        if scenario == "camera_offline" and self.cameras:
            detail["cameras"] = self.cameras.inject("camera_offline", camera_id)
        self.repo.insert_log("INFO", "simulation", f"注入场景 {scenario}", detail)
        return {"ok": True, "scenario": scenario, "detail": detail}

    @property
    def ready(self) -> bool:
        return self._ready
