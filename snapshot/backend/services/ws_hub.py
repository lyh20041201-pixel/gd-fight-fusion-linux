"""WebSocket 广播中心。

- 单向推送为主（客户端只发 ping / subscribe）。
- 每个连接一个有界队列，客户端处理不过来时丢弃最旧消息，避免后端积压。
- 前端断开不影响后端采集与报警。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)

MAX_QUEUE = 200


class _Client:
    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=MAX_QUEUE)
        self.dropped = 0

    def offer(self, payload: str) -> None:
        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:  # pragma: no cover
                pass
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:  # pragma: no cover
            self.dropped += 1


class WsHub:
    def __init__(self) -> None:
        self._clients: set[_Client] = set()
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.sent = 0

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def register(self, ws: WebSocket) -> _Client:
        client = _Client(ws)
        async with self._lock:
            self._clients.add(client)
        logger.info("WebSocket 客户端接入，当前 %d 个", len(self._clients))
        return client

    async def unregister(self, client: _Client) -> None:
        async with self._lock:
            self._clients.discard(client)
        logger.info("WebSocket 客户端断开，当前 %d 个", len(self._clients))

    # ---------- 广播 ----------
    def _encode(self, event: str, data: Any) -> str:
        return json.dumps(
            {"type": event, "ts": time.time(), "data": data},
            ensure_ascii=False,
            default=str,
        )

    def broadcast(self, event: str, data: Any) -> None:
        """线程安全的广播入口（可从采集线程调用）。"""
        payload = self._encode(event, data)
        if self._loop is None or self._loop.is_closed():
            return
        try:
            self._loop.call_soon_threadsafe(self._fanout, payload)
        except RuntimeError:  # pragma: no cover - loop 关闭中
            pass

    def _fanout(self, payload: str) -> None:
        self.sent += 1
        for client in list(self._clients):
            client.offer(payload)

    async def broadcast_async(self, event: str, data: Any) -> None:
        self._fanout(self._encode(event, data))

    async def send_to(self, client: _Client, event: str, data: Any) -> None:
        await client.ws.send_text(self._encode(event, data))
