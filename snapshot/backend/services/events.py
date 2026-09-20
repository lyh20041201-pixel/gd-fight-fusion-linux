"""风险事件服务：事件建档、关键帧落盘、Qwen 复核结果回写。

关键约束：
- 报警执行不等待图片保存与 Qwen（本函数只在报警指令下发之后被调用）；
- 图片文件存 data/events/，数据库只存路径与元数据；
- 传感器触发的事件（烟感 / CO₂ 严重超标）会从所有在线摄像头抓取现场关键帧。
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from ..ai.qwen_reviewer import QwenResult
from ..ai.queue import QwenQueue, ReviewTask
from ..audio.capture import save_audio
from ..cameras.manager import CameraManager, CameraWorker
from ..config.defaults import QWEN_REVIEW_EVENT_TYPES
from ..config.runtime import RuntimeConfig
from ..database.repositories import Repository
from ..rules.engine import RuleOutcome
from ..schemas.enums import EVENT_TYPE_LABELS, EventSource, EventStatus, RiskLevel
from ..schemas.events import EventFrameOut, QwenAnalysisOut, RiskEventOut
from ..vision.annotator import encode_jpeg

logger = logging.getLogger(__name__)

POST_FRAME_DELAY = 2.5
PRE_FRAME_OFFSET = 3.0



# 事件图片按 events_dir/YYYYMMDD/<event_id>/ 存放
_DATE_DIR = re.compile(r"^\d{8}$")


def purge_event_images(
    events_dir: Path, relative_paths: Iterable[str], retention_days: int
) -> int:
    """删除过期事件的图片，并清掉整个过期日期目录。

    日期目录整体删除是为了带走崩溃 / 写库失败留下的孤儿文件——这些文件没有任何
    数据库记录指向它们，只按路径列表删是清不掉的。
    """
    root = Path(events_dir)
    if not root.exists():
        return 0
    resolved_root = root.resolve()
    deleted = 0
    for relative in relative_paths:
        target = (root / relative).resolve()
        # 路径来自数据库，仍然按不可信输入处理
        if not target.is_relative_to(resolved_root) or not target.is_file():
            continue
        try:
            target.unlink()
            deleted += 1
        except OSError as exc:  # noqa: PERF203 - 单个文件失败不应中断整轮清理
            logger.warning("删除事件图片失败 %s: %s", relative, exc)

    cutoff_day = time.strftime(
        "%Y%m%d", time.localtime(time.time() - max(1, int(retention_days)) * 86400)
    )
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not _DATE_DIR.match(child.name) or child.is_symlink() or not child.resolve().is_relative_to(resolved_root):
            continue
        if child.name < cutoff_day:
            shutil.rmtree(child, ignore_errors=True)
        else:
            _prune_empty_dirs(child)
    return deleted


def _prune_empty_dirs(folder: Path) -> None:
    for child in folder.iterdir():
        if child.is_dir() and not child.is_symlink():
            _prune_empty_dirs(child)
    try:
        next(folder.iterdir())
    except StopIteration:
        folder.rmdir()
    except OSError:
        pass


class RiskEventService:
    def __init__(
        self,
        repo: Repository,
        cameras: CameraManager,
        runtime: RuntimeConfig,
        events_dir: Path,
        *,
        broadcast=None,
        qwen_queue: QwenQueue | None = None,
        audio=None,
    ) -> None:
        self.repo = repo
        self.cameras = cameras
        self.runtime = runtime
        self.events_dir = Path(events_dir)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self._broadcast = broadcast
        self.qwen_queue = qwen_queue
        self.audio = audio
        self._tasks: set[asyncio.Task] = set()

    # ---------- 创建 ----------
    async def create_from_outcome(
        self, outcome: RuleOutcome, context: dict[str, Any]
    ) -> RiskEventOut:
        now = time.time()
        event_id = f"EVT-{time.strftime('%Y%m%d-%H%M%S', time.localtime(now))}-{uuid.uuid4().hex[:6]}"
        event = RiskEventOut(
            event_id=event_id,
            event_type=outcome.event_type,
            event_type_label=EVENT_TYPE_LABELS.get(outcome.event_type, outcome.event_type),
            risk_level=RiskLevel(outcome.risk_level),
            source=EventSource(outcome.source),
            camera_id=outcome.camera_id,
            track_ids=list(outcome.track_ids),
            occurred_at=now,
            status=EventStatus.PENDING,
            title=outcome.title,
            description=outcome.description,
            sensor_snapshot=context.get("sensors", {}),
            track_summary=context.get("track_summary", {}),
            rule_basis={**outcome.basis, "rule_id": outcome.rule_id},
            alert_result=context.get("alert_result", {}),
            created_at=now,
            updated_at=now,
        )
        await asyncio.to_thread(self.repo.insert_event, event)
        self._emit("risk_event_created", event.model_dump(mode="json"))
        logger.warning(
            "风险事件 %s [%s/%s] %s", event_id, outcome.event_type, outcome.risk_level.value,
            outcome.description,
        )

        # 关键帧与 Qwen 复核在后台进行，绝不阻塞报警链路
        task = asyncio.create_task(
            self._capture_and_review(event, outcome, context), name=f"event-{event_id}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return event

    # ---------- 关键帧 ----------
    async def _capture_and_review(
        self, event: RiskEventOut, outcome: RuleOutcome, context: dict[str, Any]
    ) -> None:
        try:
            targets = self._target_workers(outcome)
            if not targets:
                logger.info("事件 %s 无可用摄像头，跳过关键帧保存", event.event_id)
            else:
                await self._save_frames(event, targets, role="pre", offset=PRE_FRAME_OFFSET)
                trigger_paths = await self._save_frames(event, targets, role="trigger")
                if trigger_paths:
                    event.raw_image, event.annotated_image = trigger_paths[0]
                    await asyncio.to_thread(
                        self.repo.update_event,
                        event.event_id,
                        raw_image=event.raw_image,
                        annotated_image=event.annotated_image,
                    )
                    self._emit(
                        "risk_event_updated",
                        {
                            "event_id": event.event_id,
                            "raw_image": event.raw_image,
                            "annotated_image": event.annotated_image,
                        },
                    )
            post_delay = self.audio.settings.audio_post_seconds if self.audio else POST_FRAME_DELAY
            await asyncio.sleep(max(0, event.occurred_at + post_delay - time.time()))
            await self._save_frames(event, targets, role="post")

            evidence = None
            if self.audio:
                start = event.occurred_at - self.audio.settings.audio_pre_seconds
                end = event.occurred_at + post_delay
                evidence = self.audio.evidence(start, end, outcome.camera_id)
                folder = self.events_dir / time.strftime('%Y%m%d', time.localtime(event.occurred_at)) / event.event_id
                metadata = await asyncio.to_thread(save_audio, evidence, folder, self.events_dir)
                event.rule_basis['audio'] = metadata
                await asyncio.to_thread(self.repo.update_event, event.event_id, rule_basis=event.rule_basis)
                self._emit('risk_event_updated', {'event_id': event.event_id, 'rule_basis': event.rule_basis})

            if self.qwen_queue is not None and self._needs_qwen(outcome):
                images, annotated, manifest = await asyncio.to_thread(self._collect_review_images, event)
                review_context = self._build_qwen_context(event, context)
                review_context['image_manifest'] = manifest
                if evidence:
                    review_context['audio'] = event.rule_basis['audio']
                self.qwen_queue.submit(
                    ReviewTask(
                        event_id=event.event_id,
                        context=review_context,
                        images=images,
                        annotated_images=annotated,
                        audio_wav=evidence.wav if evidence else None,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - 事件后处理失败不得影响主流程
            logger.exception("事件 %s 关键帧/复核处理失败", event.event_id)

    def _target_workers(self, outcome: RuleOutcome) -> list[CameraWorker]:
        if outcome.camera_id:
            worker = self.cameras.get(outcome.camera_id)
            if worker is not None:
                return [worker]
        # 传感器/融合类事件：抓取所有在线摄像头现场画面
        return [w for w in self.cameras.workers.values() if w.online]

    async def _save_frames(
        self,
        event: RiskEventOut,
        workers: list[CameraWorker],
        *,
        role: str,
        offset: float = 0.0,
    ) -> list[tuple[str, str | None]]:
        saved: list[tuple[str, str | None]] = []
        for worker in workers:
            data = await asyncio.to_thread(self._extract_images, worker, role, offset, event.occurred_at)
            if data is None:
                continue
            raw_image, annotated_image, captured_at = data
            paths = await asyncio.to_thread(
                self._write_images,
                event.event_id,
                worker.config.camera_id,
                role,
                raw_image,
                annotated_image,
            )
            if paths is None:
                continue
            raw_path, annotated_path = paths
            frame = EventFrameOut(
                event_id=event.event_id,
                camera_id=worker.config.camera_id,
                role=role,
                offset_seconds=captured_at - event.occurred_at,
                raw_path=raw_path,
                annotated_path=annotated_path,
                captured_at=captured_at,
            )
            await asyncio.to_thread(self.repo.insert_frame, frame)
            event.frames.append(frame)
            saved.append((raw_path, annotated_path))
        return saved

    @staticmethod
    def _extract_images(
        worker: CameraWorker, role: str, offset: float, reference: float | None = None
    ) -> tuple[np.ndarray, np.ndarray | None, float] | None:
        reference = reference if reference is not None else time.time()
        target = reference - offset if role == 'pre' else reference if role == 'trigger' else time.time()
        item = worker.ring.frame_before(target, 0)
        if item is None or abs(item[0] - target) > 1.0:
            return None
        ts, raw, meta = item
        return meta.get('evidence_raw', raw), meta.get('annotated'), ts

    def _write_images(
        self,
        event_id: str,
        camera_id: str,
        role: str,
        raw_image: np.ndarray,
        annotated_image: np.ndarray | None,
    ) -> tuple[str, str | None] | None:
        day = time.strftime("%Y%m%d")
        folder = self.events_dir / day / event_id
        try:
            folder.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%H%M%S")
            raw_path = folder / f"{camera_id}_{role}_{stamp}_raw.jpg"
            ann_path = folder / f"{camera_id}_{role}_{stamp}_annotated.jpg"
            if not cv2.imwrite(str(raw_path), raw_image):
                raise OSError('原图写入失败')
            if annotated_image is not None and not cv2.imwrite(str(ann_path), annotated_image):
                raise OSError('标注图写入失败')
        except Exception as exc:  # noqa: BLE001
            logger.error("保存事件图片失败: %s", exc)
            return None
        return (
            str(raw_path.relative_to(self.events_dir)).replace("\\", "/"),
            ann_path.relative_to(self.events_dir).as_posix() if annotated_image is not None else None,
        )

    # ---------- Qwen ----------
    def _needs_qwen(self, outcome: RuleOutcome) -> bool:
        if not self.runtime.qwen_enabled:
            return False
        if outcome.hard_alarm:
            return False  # 硬报警不依赖 Qwen
        return outcome.needs_qwen or outcome.event_type in QWEN_REVIEW_EVENT_TYPES

    def _collect_review_images(self, event: RiskEventOut):
        """Send the exact persisted evidence, with paired annotations and real capture times."""
        images, annotated, manifest = [], [], []
        frames = sorted(event.frames, key=lambda f: f.captured_at)
        limit = self.runtime.settings.qwen_max_frames
        if len(frames) > limit:
            frames = [frames[i] for i in np.linspace(0, len(frames)-1, limit, dtype=int)]
        for frame in frames:
            if not frame.raw_path:
                continue
            images.append((self.events_dir / frame.raw_path).read_bytes())
            annotated.append((self.events_dir / frame.annotated_path).read_bytes() if frame.annotated_path else None)
            manifest.append({'frame_index': len(images)-1, 'camera_id': frame.camera_id,
                             'role': frame.role, 'captured_at': frame.captured_at,
                             'annotation_available': bool(frame.annotated_path)})
        return images, annotated, manifest

    def _build_qwen_context(self, event: RiskEventOut, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "local_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.occurred_at)),
            "safety_state": context.get("safety_state"),
            "event_type": str(event.event_type),
            "event_type_label": event.event_type_label,
            "risk_level": event.risk_level.value,
            "rule_basis": event.rule_basis,
            "occupancy": context.get("occupancy"),
            "per_camera": context.get("per_camera", {}),
            "sensors": event.sensor_snapshot,
            "track_summary": event.track_summary,
            "camera_summary": context.get("camera_summary", {}),
        }

    async def store_qwen_result(self, event_id: str, result: QwenResult) -> None:
        payload = result.payload or {}
        analysis = QwenAnalysisOut(
            event_id=event_id,
            model=result.model or self.runtime.qwen_model,
            status=result.status,
            event_type=payload.get("event_type"),
            risk_level=payload.get("risk_level"),
            summary=payload.get("summary"),
            evidence=payload.get("evidence", []),
            related_track_ids=payload.get("related_track_ids", []),
            recommended_actions=payload.get("recommended_actions", []),
            needs_human_review=payload.get("needs_human_review"),
            audio_assessment=payload.get('audio_assessment'),
            error=result.error,
            latency_ms=round(result.latency_ms, 1),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            estimated_cost=result.estimated_cost(),
            created_at=time.time(),
        )
        await asyncio.to_thread(self.repo.insert_qwen, analysis, result.raw)
        self._emit(
            "risk_event_updated",
            {"event_id": event_id, "qwen": analysis.model_dump(mode="json")},
        )
        logger.info("Qwen 复核完成 %s: %s", event_id, result.status)

    # ---------- 人工处理 ----------
    async def patch_event(
        self,
        event_id: str,
        *,
        status: str | None = None,
        note: str | None = None,
        operator: str = "local",
    ) -> RiskEventOut | None:
        event = await asyncio.to_thread(self.repo.get_event, event_id)
        if event is None:
            return None
        if status:
            await asyncio.to_thread(self.repo.update_event, event_id, status=status)
            await asyncio.to_thread(
                self.repo.insert_operator_action, event_id, f"status:{status}", operator, note
            )
        if note is not None:
            await asyncio.to_thread(self.repo.update_event, event_id, note=note)
            if not status:
                await asyncio.to_thread(
                    self.repo.insert_operator_action, event_id, "note", operator, note
                )
        updated = await asyncio.to_thread(self.repo.get_event, event_id)
        if updated is not None:
            self._emit("risk_event_updated", updated.model_dump(mode="json"))
        return updated

    # ---------- 工具 ----------
    def _emit(self, event: str, data: Any) -> None:
        if self._broadcast:
            try:
                self._broadcast(event, data)
            except Exception:  # noqa: BLE001
                logger.exception("事件广播失败")

    def image_path(self, relative: str) -> Path:
        return self.events_dir / relative

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
