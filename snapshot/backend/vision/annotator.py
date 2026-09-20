"""标注画面绘制。

标注内容：人员框、匿名 Track ID、置信度、当前人数、摄像头名称、FPS、
ROI 区域、风险目标红框、风险类型与时间。

注意：仅使用检测/跟踪产生的坐标，绝不接受 Qwen 生成的框。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .tracker import Track

COLOR_NORMAL = (120, 200, 120)  # BGR 绿：普通人员
COLOR_OUT_ROI = (170, 170, 170)  # 灰：ROI 之外
COLOR_RISK = (60, 60, 235)  # 红：风险目标
COLOR_ROI = (200, 170, 60)  # 蓝青：ROI 边界
COLOR_PANEL = (28, 28, 30)
COLOR_TEXT = (238, 238, 238)

_FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
]
_pil_font_cache: dict[int, Any] = {}
_pil_available: bool | None = None


def _get_font(size: int):
    global _pil_available
    if _pil_available is False:
        return None
    try:
        from PIL import ImageFont  # type: ignore
    except ImportError:
        _pil_available = False
        return None
    if size in _pil_font_cache:
        return _pil_font_cache[size]
    for path in _FONT_CANDIDATES:
        if path.exists():
            try:
                font = ImageFont.truetype(str(path), size)
                _pil_font_cache[size] = font
                _pil_available = True
                return font
            except Exception:  # noqa: BLE001
                continue
    _pil_available = False
    return None


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int] = COLOR_TEXT,
    size: int = 14,
) -> np.ndarray:
    """支持中文的文本绘制（有中文字体时用 PIL，否则退化为 OpenCV 英文）。"""
    if _has_cjk(text):
        font = _get_font(size)
        if font is not None:
            from PIL import Image, ImageDraw  # type: ignore

            pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            draw.text(
                (origin[0], origin[1] - size),
                text,
                font=font,
                fill=(color[2], color[1], color[0]),
            )
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        size / 30.0,
        color,
        1,
        cv2.LINE_AA,
    )
    return image


def blur_faces(image: np.ndarray, tracks: Sequence[Track]) -> np.ndarray:
    """人脸模糊：对人员框上部区域做强模糊。不做任何人脸识别或身份关联。"""
    out = image
    for track in tracks:
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        h = max(1, y2 - y1)
        fy2 = y1 + int(h * 0.28)
        x1, y1 = max(0, x1), max(0, y1)
        x2, fy2 = min(out.shape[1], x2), min(out.shape[0], fy2)
        if x2 - x1 < 4 or fy2 - y1 < 4:
            continue
        roi = out[y1:fy2, x1:x2]
        ksize = max(9, (min(roi.shape[:2]) // 2) * 2 + 1)
        out[y1:fy2, x1:x2] = cv2.GaussianBlur(roi, (ksize, ksize), 0)
    return out


def draw_roi(image: np.ndarray, roi: Sequence[Sequence[float]]) -> np.ndarray:
    if not roi or len(roi) < 3:
        return image
    h, w = image.shape[:2]
    points = np.array([[int(p[0] * w), int(p[1] * h)] for p in roi], dtype=np.int32)
    overlay = image.copy()
    cv2.fillPoly(overlay, [points], (60, 50, 20))
    cv2.addWeighted(overlay, 0.18, image, 0.82, 0, image)
    cv2.polylines(image, [points], True, COLOR_ROI, 1, cv2.LINE_AA)
    return image


def annotate(
    image: np.ndarray,
    tracks: Sequence[Track],
    *,
    camera_id: str,
    camera_name: str,
    person_count: int,
    fps: float,
    inference_ms: float = 0.0,
    roi: Sequence[Sequence[float]] | None = None,
    risk_track_ids: Sequence[int] = (),
    in_roi_map: dict[int, bool] | None = None,
    risk_label: str | None = None,
    show_boxes: bool = True,
    show_ids: bool = True,
    show_roi: bool = True,
    face_blur: bool = False,
    timestamp: float | None = None,
) -> np.ndarray:
    canvas = image.copy()
    in_roi_map = in_roi_map or {}
    risk_ids = set(risk_track_ids)

    if face_blur:
        canvas = blur_faces(canvas, tracks)
    if show_roi and roi:
        canvas = draw_roi(canvas, roi)

    if show_boxes:
        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            inside = in_roi_map.get(track.track_id, True)
            color = COLOR_RISK if track.track_id in risk_ids else (
                COLOR_NORMAL if inside else COLOR_OUT_ROI
            )
            thickness = 2 if track.track_id in risk_ids else 1
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            # 底部中心点：区域归属判定依据
            bx, by = track.bottom_center
            cv2.circle(canvas, (int(bx), int(by)), 3, color, -1)
            if show_ids:
                label = f"{track.label} {track.score:.2f}"
                cv2.rectangle(
                    canvas, (x1, max(0, y1 - 16)), (x1 + 9 * len(label), y1), color, -1
                )
                cv2.putText(
                    canvas,
                    label,
                    (x1 + 3, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (20, 20, 20),
                    1,
                    cv2.LINE_AA,
                )

    # 顶部信息条
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, 26), COLOR_PANEL, -1)
    ts = timestamp or time.time()
    time_text = time.strftime("%H:%M:%S", time.localtime(ts))
    header = f"{camera_id}  {camera_name}"
    canvas = draw_text(canvas, header, (8, 19), size=15)
    right = f"P:{person_count}  FPS:{fps:.1f}  {inference_ms:.0f}ms  {time_text}"
    cv2.putText(
        canvas,
        right,
        (max(8, w - 9 * len(right) - 8), 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )

    if risk_label:
        cv2.rectangle(canvas, (0, h - 26), (w, h), (30, 30, 120), -1)
        canvas = draw_text(canvas, f"风险: {risk_label}", (8, h - 7), color=(240, 240, 255), size=15)
    return canvas


# COCO-17: face, shoulders/arms, torso and legs (zero-based joint indices).
POSE_LINKS = ((0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(5,6),
              (5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
              (11,13),(13,15),(12,14),(14,16))


def draw_skeletons(image: np.ndarray, people, confidence: float = .3) -> np.ndarray:
    """Draw measured joints only, without changing the source/evidence frame."""
    canvas = image.copy()
    h, w = canvas.shape[:2]
    for person in people:
        joints = np.asarray(person, dtype=np.float32)
        if joints.shape != (17, 3):
            continue
        valid = (np.isfinite(joints).all(axis=1) & (joints[:, 2] >= confidence)
                 & (joints[:, 0] > 0) & (joints[:, 0] < w)
                 & (joints[:, 1] > 0) & (joints[:, 1] < h))
        points = {i: tuple(np.rint(joints[i, :2]).astype(int)) for i in np.flatnonzero(valid)}
        for a, b in POSE_LINKS:
            if a not in points or b not in points:
                continue
            color = (60, 215, 255) if a in (1,3,5,7,9,11,13,15) else (255, 215, 55)
            cv2.line(canvas, points[a], points[b], (25, 25, 25), 4, cv2.LINE_AA)
            cv2.line(canvas, points[a], points[b], color, 2, cv2.LINE_AA)
        for point in points.values():
            cv2.circle(canvas, point, 4, (25, 25, 25), -1, cv2.LINE_AA)
            cv2.circle(canvas, point, 2, (120, 255, 160), -1, cv2.LINE_AA)
    return canvas


def encode_jpeg(image: np.ndarray, quality: int = 75) -> bytes | None:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return buffer.tobytes()


def placeholder_frame(
    width: int, height: int, text: str, sub_text: str = ""
) -> np.ndarray:
    """摄像头掉线时的占位画面。"""
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.rectangle(canvas, (12, 12), (width - 12, height - 12), (60, 60, 64), 1)
    canvas = draw_text(canvas, text, (28, height // 2), color=(180, 180, 190), size=18)
    if sub_text:
        canvas = draw_text(
            canvas, sub_text, (28, height // 2 + 26), color=(130, 130, 140), size=14
        )
    return canvas
