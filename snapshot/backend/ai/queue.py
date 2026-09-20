"""Qwen 异步复核任务队列。

- 有界队列：满时丢弃最旧任务并计数，绝不阻塞规则引擎；
- 单 worker 串行执行，天然限流；
- 任何异常都被捕获，Qwen 失败不会影响本地检测与报警。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .qwen_reviewer import QwenResult, QwenReviewer

logger = logging.getLogger(__name__)


@dataclass
class ReviewTask:
    event_id: str
    context: dict[str, Any]
    images: list[bytes] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    audio_wav: bytes | None = None
    annotated_images: list[bytes | None] = field(default_factory=list)


class QwenQueue:
    def __init__(
        self,
        reviewer: QwenReviewer,
        on_result: Callable[[str, QwenResult], Awaitable[None]],
        *,
        maxsize: int = 32,
    ) -> None:
        self.reviewer = reviewer
        self._on_result = on_result
        self._queue: asyncio.Queue[ReviewTask] = asyncio.Queue(maxsize=maxsize)
        self._worker: asyncio.Task | None = None
        self.dropped = 0
        self.processed = 0
        self.pending_events: set[str] = set()
        self._notifications: set[asyncio.Task] = set()

    def _failed(self, task: ReviewTask, reason: str) -> None:
        async def notify():
            try:
                await self._on_result(task.event_id, QwenResult("failed", error=reason))
            except Exception:
                logger.exception("复核失败状态回写失败 %s", task.event_id)
        t = asyncio.create_task(notify())
        self._notifications.add(t)
        t.add_done_callback(self._notifications.discard)

    async def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run(), name="qwen-queue")

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        while not self._queue.empty():
            task = self._queue.get_nowait()
            self.pending_events.discard(task.event_id)
            self._queue.task_done()
            self._failed(task, "复核队列关闭")
        if self._notifications:
            await asyncio.gather(*list(self._notifications), return_exceptions=True)

    def submit(self, task: ReviewTask) -> bool:
        if task.event_id in self.pending_events:
            return False
        if self._queue.full():
            try:
                dropped = self._queue.get_nowait()
                self.pending_events.discard(dropped.event_id)
                self.dropped += 1
                self._queue.task_done()
                self._failed(dropped, "复核队列已满，候选被移出，待人工核查")
                logger.warning("Qwen 队列已满，丢弃最旧任务 %s", dropped.event_id)
            except asyncio.QueueEmpty:  # pragma: no cover
                pass
        try:
            self._queue.put_nowait(task)
            self.pending_events.add(task.event_id)
            return True
        except asyncio.QueueFull:  # pragma: no cover
            self.dropped += 1
            return False

    async def _run(self) -> None:
        while True:
            task = await self._queue.get()
            try:
                if task.audio_wav is not None or task.annotated_images or 'audio' in task.context:
                    result = await self.reviewer.review(task.context, task.images,
                        audio_wav=task.audio_wav, annotated_images=task.annotated_images)
                else:
                    result = await self.reviewer.review(task.context, task.images)
                self.processed += 1
                await self._on_result(task.event_id, result)
            except asyncio.CancelledError:
                self._failed(task, "复核执行被取消")
                raise
            except Exception:  # noqa: BLE001 - 队列必须永不退出
                logger.exception("Qwen 复核任务处理失败: %s", task.event_id)
                self._failed(task, "复核执行异常")
            finally:
                self.pending_events.discard(task.event_id)
                self._queue.task_done()

    def stats(self) -> dict[str, Any]:
        return {
            "queued": self._queue.qsize(),
            "processed": self.processed,
            "dropped": self.dropped,
            "running": bool(self._worker and not self._worker.done()),
        }
