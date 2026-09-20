"""真实 USB 摄像头（罗技等 UVC 设备）。"""

from __future__ import annotations

import logging
import platform
import time

import cv2

from .base import CameraSource, Frame

logger = logging.getLogger(__name__)


def _preferred_backend(backend: str = 'AUTO') -> int:
    if backend != 'AUTO':
        return {'DSHOW': cv2.CAP_DSHOW, 'MSMF': cv2.CAP_MSMF}[backend]
    if platform.system() == "Windows":
        return cv2.CAP_DSHOW  # Windows 上 DirectShow 打开多路 USB 摄像头更稳定
    return cv2.CAP_ANY


def probe_cameras(max_index: int = 6, backend: str = 'AUTO') -> list[dict[str, object]]:
    """探测本机可用摄像头索引（真实模式设置页使用）。"""
    found: list[dict[str, object]] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, _preferred_backend(backend))
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                found.append(
                    {
                        "index": index,
                        "available": bool(ok),
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    }
                )
        finally:
            cap.release()
    return found


class UsbCamera(CameraSource):
    def __init__(
        self,
        camera_id: str,
        name: str,
        index: int,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        reconnect_seconds: float = 3.0,
        backend: str = 'AUTO',
        fourcc: str = 'AUTO',
    ) -> None:
        super().__init__(camera_id, name, f"usb:{index}")
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.reconnect_seconds = reconnect_seconds
        self.backend = backend
        self.fourcc = fourcc
        self.capture_info: dict[str, object] = {}
        self._cap: cv2.VideoCapture | None = None
        self._frame_index = 0
        self._next_retry = 0.0
        self._fail_count = 0
        self._last_read = 0.0

    def open(self) -> bool:
        now = time.time()
        if now < self._next_retry:
            return False
        cap = cv2.VideoCapture(self.index, _preferred_backend(self.backend))
        if not cap.isOpened():
            cap.release()
            self._opened = False
            self.last_error = f"无法打开摄像头 index={self.index}"
            self._next_retry = now + self.reconnect_seconds
            return False
        accepted = {}
        # Negotiate pixel format before size/rate, without changing device-wide
        # exposure, sharpening or colour controls used by other applications.
        if self.fourcc != 'AUTO':
            accepted['fourcc'] = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        accepted['width'] = cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        accepted['height'] = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        accepted['fps'] = cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 只保留最新帧，避免延迟累积
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc = ''.join(chr((code >> (8 * i)) & 255) for i in range(4)).strip('\x00')
        self.capture_info = dict(backend=cap.getBackendName(), fourcc=actual_fourcc or 'unknown',
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS), requested_backend=self.backend,
            requested_fourcc=self.fourcc, requested_width=self.width, requested_height=self.height,
            requested_fps=self.fps, fourcc_code=code, accepted=accepted)
        if not actual_fourcc.isprintable():
            self.capture_info['fourcc'] = f'backend-code:{code}'
        if self.fourcc != 'AUTO' and (not accepted['fourcc'] or actual_fourcc != self.fourcc):
            self.capture_info['warning'] = f'设备未确认请求的 {self.fourcc} 格式，请核对采集状态'
            logger.warning('摄像头 %s %s', self.camera_id, self.capture_info['warning'])
        self._cap = cap
        self._opened = True
        self.last_error = None
        self._fail_count = 0
        logger.info("摄像头 %s 已打开 (index=%d): %s", self.camera_id, self.index, self.capture_info)
        return True

    def read(self) -> Frame | None:
        if self._cap is None or not self._opened:
            if not self.open():
                return None
        assert self._cap is not None
        # 按配置帧率节流：摄像头本身可能以 30 FPS 输出，没必要每帧都推理
        interval = 1.0 / max(1, self.fps)
        wait = self._last_read + interval - time.time()
        if wait > 0:
            time.sleep(min(wait, interval))
        self._last_read = time.time()
        try:
            ok, image = self._cap.read()
        except cv2.error as exc:  # pragma: no cover - 驱动异常
            ok, image = False, None
            self.last_error = f"读取失败: {exc}"
        if not ok or image is None:
            self._fail_count += 1
            self.last_error = self.last_error or "读取帧失败"
            if self._fail_count >= 5:
                logger.warning("摄像头 %s 连续读取失败，触发重连", self.camera_id)
                self.close()
                self._next_retry = time.time() + self.reconnect_seconds
            return None
        self._fail_count = 0
        self._frame_index += 1
        return Frame(image=image, timestamp=time.time(), index=self._frame_index, meta={})

    def close(self) -> None:
        self._opened = False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
            self._cap = None
