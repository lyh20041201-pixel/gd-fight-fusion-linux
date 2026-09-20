"""摄像头管理与视觉流水线。

每一路摄像头一个独立工作线程：采集 -> 检测 -> 跟踪 -> ROI 判定 -> 标注 -> 有界缓冲。
- 单路失败只影响自身（自动重连），不拖垮其他摄像头；
- 只保留最新帧，处理不过来时直接丢弃旧帧，杜绝延迟累积；
- 每路维护约 5 秒关键帧环形缓冲，用于事件前后帧回溯。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

from ..config.runtime import RuntimeConfig
from ..config.settings import Settings
from ..config.vision import VisionConfig
from ..fusion.occupancy import OccupancyFusion
from ..fusion.regions import default_roi_for_index, in_roi
from ..schemas.common import CameraOut, ModuleStatus, TrackBox
from ..schemas.enums import ModuleState, ModuleType
from ..vision.annotator import annotate, blur_faces, draw_skeletons, encode_jpeg, placeholder_frame
from ..vision.detector import BaseDetector
from ..vision.tracker import ByteTracker, Track
from .base import CameraSource
from .ring_buffer import FrameRingBuffer, LatestFrameSlot
from .sim_source import SimulatedCamera
from .usb_source import UsbCamera

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    camera_id: str
    name: str
    region: str
    is_primary_overlap: bool = True
    roi: list[list[float]] = field(default_factory=list)
    enabled: bool = True


class CameraWorker:
    """单路摄像头的采集与视觉处理线程。"""

    def __init__(
        self,
        source: CameraSource,
        config: CameraConfig,
        detector: BaseDetector,
        detector_lock: threading.Lock,
        *,
        buffer_seconds: float = 5.0,
        face_blur: bool = False,
        obstruction_variance: float = 40.0,
        obstruction_hold_seconds: float = 3.0,
        tracker: ByteTracker | None = None,
        action_sample_interval: float = 0.2,
    ) -> None:
        self.source = source
        self.config = config
        self.detector = detector
        self._detector_lock = detector_lock
        self.tracker = tracker or ByteTracker(camera_id=config.camera_id)
        self.ring = FrameRingBuffer(seconds=max(5.0, buffer_seconds), sample_interval=action_sample_interval)
        self.raw_slot = LatestFrameSlot()
        self.annotated_slot = LatestFrameSlot()

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()

        self.online = False
        self.person_count = 0
        self.fps = 0.0
        self.inference_ms = 0.0
        self.last_frame_at: float | None = None
        self.fault_reason: str | None = None
        self.tracks: list[Track] = []
        self.in_roi_map: dict[int, bool] = {}
        self.risk_track_ids: set[int] = set()
        self.risk_label: str | None = None
        self.obstructed = False
        self._obstruction_since: float | None = None
        self._face_blur = face_blur
        self._obstruction_variance = obstruction_variance
        self._obstruction_hold = obstruction_hold_seconds
        self._frame_count = 0
        self._fps_window: list[float] = []
        self._last_jpeg_raw: bytes | None = None
        self._last_jpeg_annotated: bytes | None = None
        self._pose_preview: tuple[float, bytes] | None = None

    # ---------- 生命周期 ----------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"cam-{self.config.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        try:
            self.source.close()
        except Exception:  # noqa: BLE001
            logger.warning("释放摄像头 %s 失败", self.config.camera_id)

    # ---------- 主循环 ----------
    def _run(self) -> None:
        if not self.source.opened:
            self.source.open()
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:  # noqa: BLE001 - 单路异常不得影响其他摄像头
                logger.exception("摄像头 %s 处理异常", self.config.camera_id)
                self.fault_reason = "处理异常，正在恢复"
                time.sleep(0.5)

    def _tick(self) -> None:
        frame = self.source.read()
        if frame is None:
            if self.online:
                logger.warning("摄像头 %s 掉线: %s", self.config.camera_id, self.source.last_error)
            self.online = False
            self.fault_reason = self.source.last_error or "无法读取画面"
            self.fps = 0.0
            self._push_placeholder()
            time.sleep(0.5)
            if not self.source.opened:
                self.source.open()
            return

        now = frame.timestamp
        self.online = True
        self.fault_reason = None
        self._frame_count += 1
        self._update_fps(now)

        image = frame.image
        height, width = image.shape[:2]
        # Video event evidence is captured even if person inference fails.
        self.ring.offer(image, now, {})

        # 把「哪一路 / 第几帧 / 什么时刻」一并传给检测器，
        # 使每个 Detection 都自带溯源信息（事件建档与轨迹摘要要用）
        meta = dict(frame.meta or {})
        meta.update(
            {
                "camera_id": self.config.camera_id,
                "frame_id": frame.index,
                "timestamp": now,
            }
        )
        started = time.perf_counter()
        with self._detector_lock:
            detections = self.detector.detect(image, meta)
        self.inference_ms = round((time.perf_counter() - started) * 1000, 2)

        tracks = self.tracker.update(detections)
        roi = self.config.roi or default_roi_for_index(0)
        roi_map = {t.track_id: in_roi(t.bbox, roi, width, height) for t in tracks}
        count = sum(1 for t in tracks if roi_map.get(t.track_id, True))

        obstructed = self._detect_obstruction(image, now)

        with self._lock:
            self.tracks = tracks
            self.in_roi_map = roi_map
            self.person_count = count
            self.last_frame_at = now
            self.obstructed = obstructed

        annotated = annotate(
            image,
            tracks,
            camera_id=self.config.camera_id,
            camera_name=self.config.name,
            person_count=count,
            fps=self.fps,
            inference_ms=self.inference_ms,
            roi=roi,
            risk_track_ids=self.risk_track_ids,
            in_roi_map=roi_map,
            risk_label=self.risk_label,
            face_blur=self._face_blur,
            timestamp=now,
        )

        self.raw_slot.put(image, now, frame.meta)
        evidence_raw = blur_faces(image.copy(), tracks) if self._face_blur else image
        self.ring.attach_evidence(now, annotated, evidence_raw)
        self.annotated_slot.put(annotated, now, {"count": count})
        self._last_jpeg_raw = encode_jpeg(image)
        self._last_jpeg_annotated = encode_jpeg(annotated)

    def _update_fps(self, now: float) -> None:
        self._fps_window.append(now)
        if len(self._fps_window) > 20:
            self._fps_window = self._fps_window[-20:]
        if len(self._fps_window) >= 2:
            span = self._fps_window[-1] - self._fps_window[0]
            if span > 0:
                self.fps = round((len(self._fps_window) - 1) / span, 1)

    def _detect_obstruction(self, image: np.ndarray, now: float) -> bool:
        """画面方差过低持续一段时间 -> 疑似被遮挡 / 失焦 / 断流。

        阈值与持续时间都来自可配置项（DEFAULT_THRESHOLDS.obstruction_variance /
        obstruction_hold_seconds），不写死在代码里。
        """
        if self._frame_count % 5 != 0:
            return self.obstructed
        gray = image.mean(axis=2) if image.ndim == 3 else image
        variance = float(np.var(gray))
        threshold = float(self._obstruction_variance)
        hold = float(self._obstruction_hold)
        if variance < threshold:
            self._obstruction_since = self._obstruction_since or now
            return now - self._obstruction_since >= hold
        self._obstruction_since = None
        return False

    def _push_placeholder(self) -> None:
        frame = placeholder_frame(
            640, 480, f"{self.config.camera_id} 离线", self.fault_reason or ""
        )
        self._last_jpeg_raw = encode_jpeg(frame)
        self._last_jpeg_annotated = self._last_jpeg_raw

    # ---------- 对外 ----------
    def set_face_blur(self, enabled: bool) -> None:
        self._face_blur = enabled

    def set_risk(self, track_ids: Sequence[int], label: str | None) -> None:
        self.risk_track_ids = set(track_ids)
        self.risk_label = label

    def clear_risk(self) -> None:
        self.risk_track_ids = set()
        self.risk_label = None

    def set_pose_preview(self, frame, result) -> None:
        timestamp, _, meta = frame
        # The base annotation and pose must be from exactly the same capture.
        if result.get('timestamp') != timestamp or 'annotated' not in meta:
            return
        if not self.online or not 0 <= time.time()-timestamp <= 1.5:
            return
        payload = encode_jpeg(draw_skeletons(meta['annotated'], result['keypoints']))
        if payload:
            with self._lock:
                self._pose_preview = (timestamp, payload)

    def jpeg(self, annotated: bool = True) -> bytes | None:
        with self._lock:
            preview = self._pose_preview
        if annotated and self.online and preview and 0 <= time.time()-preview[0] <= 1.5:
            return preview[1]
        return self._last_jpeg_annotated if annotated else self._last_jpeg_raw

    def latest_images(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        raw = self.raw_slot.get()
        ann = self.annotated_slot.get()
        return (raw[0] if raw else None, ann[0] if ann else None)

    def track_boxes(self) -> list[TrackBox]:
        with self._lock:
            return [
                TrackBox(
                    track_id=t.track_id,
                    label=t.label,
                    bbox=[round(v, 1) for v in t.bbox],
                    confidence=round(t.score, 3),
                    in_roi=self.in_roi_map.get(t.track_id, True),
                    risk=t.track_id in self.risk_track_ids,
                    age=t.age,
                )
                for t in self.tracks
            ]

    def snapshot(self) -> CameraOut:
        state = ModuleState.ONLINE if self.online else ModuleState.OFFLINE
        if self.online and self.obstructed:
            state = ModuleState.WARNING
        return CameraOut(
            camera_id=self.config.camera_id,
            name=self.config.name,
            region=self.config.region,
            source=self.source.source,
            online=self.online,
            state=state,
            person_count=self.person_count,
            fps=self.fps,
            inference_ms=self.inference_ms,
            last_frame_at=self.last_frame_at,
            roi=self.config.roi,
            is_primary_overlap=self.config.is_primary_overlap,
            fault_reason="画面疑似被遮挡" if self.obstructed else self.fault_reason,
            tracks=self.track_boxes(),
            obstructed=self.obstructed,
        )

    def track_summary(self) -> dict[str, Any]:
        summary = self.tracker.summary()
        summary.update(
            {
                "camera_id": self.config.camera_id,
                "region": self.config.region,
                "in_roi": self.person_count,
                "fps": self.fps,
                "inference_ms": self.inference_ms,
            }
        )
        return summary


class CameraManager:
    def __init__(
        self,
        settings: Settings,
        runtime: RuntimeConfig,
        detector: BaseDetector,
        fusion: OccupancyFusion,
        *,
        detector_degraded: str | None = None,
        vision_config: VisionConfig | None = None,
    ) -> None:
        self.settings = settings
        self.runtime = runtime
        self.detector = detector
        self.detector_degraded = detector_degraded
        self.vision_config = vision_config or VisionConfig()
        self.fusion = fusion
        self.workers: dict[str, CameraWorker] = {}
        self._detector_lock = threading.Lock()
        self._started = False

    # ---------- 构建 ----------
    def build(self, configs: list[CameraConfig] | None = None) -> None:
        simulation = self.runtime.simulation_mode
        indices = self.runtime.get("camera_device_indices") or self.settings.camera_indices
        # 真实模式：有几个配置的设备索引就建几路，没有配置就一路也不建（界面显示"未配置"）
        count = max(1, self.settings.simulated_camera_count) if simulation else len(indices)

        provided = {c.camera_id: c for c in (configs or [])}
        for i in range(count):
            camera_id = f"CAM-{i + 1:02d}"
            config = provided.get(camera_id) or CameraConfig(
                camera_id=camera_id,
                name=f"摄像头 {i + 1}",
                region=f"区域 {i + 1}",
                roi=default_roi_for_index(i),
            )
            if not config.roi:
                config.roi = default_roi_for_index(i)
            if not config.enabled:
                continue
            if simulation:
                source: CameraSource = SimulatedCamera(
                    camera_id,
                    config.name,
                    index=i,
                    width=self.settings.camera_width,
                    height=self.settings.camera_height,
                    fps=min(12, self.settings.camera_fps),
                    base_people=[3, 2, 4, 2][i % 4],
                )
            else:
                device_index = indices[i] if i < len(indices) else i
                source = UsbCamera(
                    camera_id,
                    config.name,
                    index=device_index,
                    width=self.settings.camera_width,
                    height=self.settings.camera_height,
                    fps=self.settings.camera_fps,
                    backend=self.settings.camera_backend,
                    fourcc=self.settings.camera_fourcc,
                )
            worker = CameraWorker(
                source,
                config,
                self.detector,
                self._detector_lock,
                buffer_seconds=self.settings.frame_buffer_seconds,
                face_blur=self.runtime.face_blur_enabled,
                obstruction_variance=float(
                    self.runtime.threshold(
                        "obstruction_variance", self.vision_config.obstruction.variance
                    )
                ),
                obstruction_hold_seconds=float(
                    self.runtime.threshold(
                        "obstruction_hold_seconds", self.vision_config.obstruction.hold_seconds
                    )
                ),
                tracker=self._build_tracker(camera_id),
                action_sample_interval=0.1 if self.settings.live_actions_enabled else 0.2,
            )
            self.workers[camera_id] = worker
            self.fusion.register(camera_id, config.region, config.is_primary_overlap)

    def _build_tracker(self, camera_id: str) -> ByteTracker:
        """每路摄像头一个独立的跟踪器实例，参数来自 config/vision.yaml。"""
        cfg = self.vision_config.tracker
        return ByteTracker(
            track_thresh=cfg.track_thresh,
            low_thresh=cfg.low_thresh,
            match_thresh=cfg.match_thresh,
            second_match_thresh=cfg.second_match_thresh,
            track_buffer=cfg.track_buffer,
            min_hits=cfg.min_hits,
            matcher=cfg.matcher,
            camera_id=camera_id,
        )

    def start(self) -> None:
        for worker in self.workers.values():
            worker.start()
        self._started = True
        logger.info("已启动 %d 路摄像头", len(self.workers))

    def rebuild(self, configs: list[CameraConfig] | None = None) -> None:
        """按最新配置重建全部摄像头（用于在界面上更换/增减接入的设备）。

        会先释放旧的采集线程与设备句柄，再按新配置重新打开，无需重启后端。
        """
        was_started = self._started
        self.stop()
        for camera_id in list(self.workers):
            self.fusion.unregister(camera_id)
        self.workers.clear()
        self.build(configs)
        if was_started:
            self.start()
        logger.info("摄像头已按新配置重建，共 %d 路", len(self.workers))

    def stop(self) -> None:
        for worker in self.workers.values():
            worker.stop()
        self._started = False
        logger.info("所有摄像头已释放")

    # ---------- 查询 ----------
    def get(self, camera_id: str) -> CameraWorker | None:
        return self.workers.get(camera_id)

    def snapshots(self) -> list[CameraOut]:
        return [w.snapshot() for w in self.workers.values()]

    def refresh_fusion(self) -> None:
        for camera_id, worker in self.workers.items():
            self.fusion.update(camera_id, worker.person_count, worker.online)

    def online_count(self) -> int:
        return sum(1 for w in self.workers.values() if w.online)

    def apply_config(self, camera_id: str, **patch: Any) -> bool:
        worker = self.workers.get(camera_id)
        if worker is None:
            return False
        config = worker.config
        if patch.get("name"):
            config.name = patch["name"]
        if patch.get("region") is not None:
            config.region = patch["region"]
        if patch.get("roi") is not None:
            config.roi = patch["roi"] or config.roi
        if patch.get("is_primary_overlap") is not None:
            config.is_primary_overlap = bool(patch["is_primary_overlap"])
        self.fusion.register(camera_id, config.region, config.is_primary_overlap)
        return True

    def set_face_blur(self, enabled: bool) -> None:
        for worker in self.workers.values():
            worker.set_face_blur(enabled)

    # ---------- 模拟场景 ----------
    def inject(self, scenario: str, camera_id: str | None = None) -> dict[str, Any]:
        targets = (
            [self.workers[camera_id]] if camera_id and camera_id in self.workers
            else list(self.workers.values())
        )
        touched: list[str] = []
        for worker in targets:
            source = worker.source
            if not isinstance(source, SimulatedCamera):
                continue
            if scenario == "camera_offline":
                source.set_offline(45)
            elif scenario == "camera_obstructed":
                source.set_obstructed(40)
            elif scenario == "over_capacity":
                source.set_people(int(self.runtime.threshold("occupancy_limit", 60)) // 2 + 6)
            elif scenario == "after_hours_presence":
                source.set_people(max(1, source.people))
            elif scenario == "recover":
                source.recover()
                source.set_people(source.base_people)
            touched.append(worker.config.camera_id)
        if not touched:
            return {"ok": False, "message": "真实摄像头不支持场景注入"}
        return {"ok": True, "cameras": touched, "scenario": scenario}

    # ---------- MJPEG ----------
    def mjpeg_stream(self, camera_id: str, annotated: bool = True) -> Iterator[bytes]:
        worker = self.workers.get(camera_id)
        interval = 1.0 / max(1, self.settings.stream_fps)
        boundary = b"--frame\r\n"
        while True:
            start = time.time()
            if worker is None:
                frame = placeholder_frame(640, 480, f"{camera_id} 不存在")
                payload = encode_jpeg(frame)
            else:
                payload = worker.jpeg(annotated)
                if payload is None:
                    frame = placeholder_frame(
                        640, 480, f"{camera_id} 无画面", worker.fault_reason or "等待首帧"
                    )
                    payload = encode_jpeg(frame)
            if payload:
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
            elapsed = time.time() - start
            time.sleep(max(0.0, interval - elapsed))

    def snapshot_jpeg(self, camera_id: str, annotated: bool = True) -> bytes | None:
        worker = self.workers.get(camera_id)
        if worker is None:
            return encode_jpeg(placeholder_frame(640, 480, f"{camera_id} 不存在"))
        payload = worker.jpeg(annotated)
        if payload is None:
            return encode_jpeg(
                placeholder_frame(640, 480, f"{camera_id} 无画面", worker.fault_reason or "")
            )
        return payload

    # ---------- 模块状态 ----------
    def module_statuses(self) -> list[ModuleStatus]:
        modules: list[ModuleStatus] = []
        for worker in self.workers.values():
            snap = worker.snapshot()
            modules.append(
                ModuleStatus(
                    module_id=snap.camera_id,
                    name=f"{snap.name}（{snap.region}）",
                    module_type=ModuleType.CAMERA,
                    state=snap.state,
                    value_text=(
                        f"{snap.person_count} 人 · {snap.fps:.1f} FPS"
                        if snap.online
                        else "离线"
                    ),
                    detail={
                        "region": snap.region,
                        "source": snap.source,
                        "inference_ms": snap.inference_ms,
                        "is_primary_overlap": snap.is_primary_overlap,
                        "obstructed": snap.obstructed,
                        "tracks": len(snap.tracks),
                    },
                    last_update=snap.last_frame_at,
                    fault_reason=snap.fault_reason,
                    actions=["reconnect"],
                )
            )
        return modules

    def vision_module_statuses(self) -> list[ModuleStatus]:
        now = time.time()
        degraded = self.detector_degraded
        backend = self.detector.backend
        total_tracks = sum(len(w.tracks) for w in self.workers.values())
        avg_infer = (
            round(
                sum(w.inference_ms for w in self.workers.values()) / max(1, len(self.workers)),
                1,
            )
            if self.workers
            else 0.0
        )
        # 模拟模式使用模拟检测器属于预期行为，不算降级；真实模式回退才是降级
        simulated_expected = self.runtime.simulation_mode and backend == "simulated"
        detector_state = (
            ModuleState.ONLINE if (not degraded or simulated_expected) else ModuleState.WARNING
        )
        if backend == "yolo":
            detector_text = f"YOLO 推理中 · 平均 {avg_infer} ms"
        elif backend == "simulated":
            detector_text = f"模拟检测器 · 平均 {avg_infer} ms"
        else:
            detector_text = "未启用人员检测"

        return [
            ModuleStatus(
                module_id="service.yolo",
                name="YOLO 人员检测服务",
                module_type=ModuleType.SERVICE,
                state=detector_state,
                value_text=detector_text,
                detail={
                    "backend": backend,
                    "degraded": degraded,
                    "simulated_expected": simulated_expected,
                },
                last_update=now,
                fault_reason=None if simulated_expected else degraded,
            ),
            ModuleStatus(
                module_id="service.bytetrack",
                name="ByteTrack 跟踪服务",
                module_type=ModuleType.SERVICE,
                state=ModuleState.ONLINE,
                value_text=f"活跃轨迹 {total_tracks}",
                detail={
                    camera_id: worker.tracker.summary()
                    for camera_id, worker in self.workers.items()
                },
                last_update=now,
            ),
        ]
