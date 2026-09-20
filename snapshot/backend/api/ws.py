"""WebSocket 实时推送。

连接建立后先下发一次全量快照（snapshot），随后持续推送增量事件。
客户端可发送 {"type":"ping"} 保活；前端负责断线自动重连。
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config.settings import get_settings
from ..services.state import AppState

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/realtime")
async def realtime(websocket: WebSocket) -> None:
    state: AppState | None = getattr(websocket.app.state, "app_state", None)
    # WebSocket 握手不受同源策略与 CORS 约束：任意网页都能连上本机服务并读取
    # 全量快照，因此必须在 accept 之前校验 Origin。无 Origin 头的是非浏览器
    # 客户端（本机脚本、测试），放行。
    origin = websocket.headers.get("origin")
    if origin is not None and origin.rstrip("/") not in get_settings().allowed_origins:
        logger.warning("拒绝来源不合法的 WebSocket 连接: %s", origin)
        await websocket.close(code=1008)
        return
    await websocket.accept()
    if state is None or not state.ready:
        await websocket.send_text(json.dumps({"type": "error", "data": "服务未就绪"}))
        await websocket.close()
        return

    client = await state.ws.register(websocket)
    try:
        await state.ws.send_to(client, "snapshot", state.ws_snapshot())
    except Exception:  # noqa: BLE001
        logger.exception("发送初始快照失败")

    async def _sender() -> None:
        while True:
            payload = await client.queue.get()
            await websocket.send_text(payload)

    async def _receiver() -> None:
        while True:
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
            except ValueError:
                continue
            if data.get("type") == "ping":
                await state.ws.send_to(client, "pong", {"server_time": data.get("ts")})
            elif data.get("type") == "request_snapshot":
                await state.ws.send_to(client, "snapshot", state.ws_snapshot())

    sender = asyncio.create_task(_sender(), name="ws-sender")
    receiver = asyncio.create_task(_receiver(), name="ws-receiver")
    try:
        done, pending = await asyncio.wait(
            {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        logger.debug("WebSocket 连接异常结束", exc_info=True)
    finally:
        sender.cancel()
        receiver.cancel()
        await state.ws.unregister(client)
