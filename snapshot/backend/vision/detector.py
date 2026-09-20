"""人员检测器。

- YoloDetector：加载预训练 YOLO（ultralytics），只取 person 类别，不做自训练。
- SimulatedDetector：仅模拟模式使用，基于合成画面的真值框生成检测结果（含抖动、漏检）。
- NullDetector：真实模式下缺依赖/缺权重时使用，不返回任何框。

真实模式绝不会退化成模拟检测器：宁可没有检测结果，也不能在真实画面上画出假框；
此时 YOLO 模块在界面上标记为降级并给出原因。
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"


@dataclass
class Detection:
    """一次检测结果的统一表示。

    全项目只用这一个结构在「检测 -> 跟踪 -> ROI 判定 -> 绘制 -> 统计 -> 事件」
    之间传递，不再用形状各异的 dict。后面几个字段带默认值，
    既能补全溯源信息（哪一路、第几帧、什么时刻），又不影响已有调用。
    """

    bbox: list[float]  # [x1, y1, x2, y2]，像素坐标
    confidence: float
    class_id: int = PERSON_CLASS_ID
    class_name: str = PERSON_CLASS_NAME
    camera_id: str = ""
    frame_id: int = 0
    timestamp: float = 0.0

    @property
    def bottom_center(self) -> tuple[float, float]:
        """人的落脚点。ROI 判定用它，比框中心更贴合地面区域划分。"""
        x1, _y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height


class DetectorUnavailable(RuntimeError):
    pass


class BaseDetector:
    backend = "base"
    degraded_reason: str | None = None

    def detect(self, image: np.ndarray, meta: dict[str, Any] | None = None) -> list[Detection]:
        raise NotImplementedError

    @staticmethod
    def _stamp(detections: list[Detection], meta: dict[str, Any] | None) -> list[Detection]:
        """给检测结果补上「哪一路 / 第几帧 / 什么时刻」，便于事件溯源。"""
        meta = meta or {}
        camera_id = str(meta.get("camera_id", ""))
        frame_id = int(meta.get("frame_id", 0) or 0)
        timestamp = float(meta.get("timestamp", 0.0) or 0.0)
        for det in detections:
            det.camera_id = camera_id
            det.frame_id = frame_id
            det.timestamp = timestamp
        return detections

    def close(self) -> None:  # pragma: no cover - 默认无资源
        return


class SimulatedDetector(BaseDetector):
    """基于模拟画面真值的检测器（等价于"完美但有噪声"的 YOLO）。"""

    backend = "simulated"

    def __init__(self, confidence: float = 0.1, seed: int | None = 7) -> None:
        self.confidence = confidence
        self._rng = random.Random(seed)

    def detect(self, image: np.ndarray, meta: dict[str, Any] | None = None) -> list[Detection]:
        meta = meta or {}
        truth = meta.get("truth") or []
        detections: list[Detection] = []
        for item in truth:
            if self._rng.random() < 0.04:  # 模拟漏检
                continue
            x1, y1, x2, y2 = item["bbox"]
            jitter = lambda: self._rng.uniform(-4.0, 4.0)  # noqa: E731
            conf = float(item.get("confidence", 0.8))
            conf = max(0.2, min(0.98, conf + self._rng.uniform(-0.12, 0.06)))
            if conf < self.confidence:
                continue
            detections.append(
                Detection(
                    bbox=[x1 + jitter(), y1 + jitter(), x2 + jitter(), y2 + jitter()],
                    confidence=round(conf, 3),
                )
            )
        # 偶发误检，用于验证低置信度触发 Qwen 复核的链路
        if self._rng.random() < 0.01 and image is not None:
            h, w = image.shape[:2]
            x = self._rng.uniform(0, w - 60)
            y = self._rng.uniform(h * 0.4, h - 90)
            detections.append(
                Detection(bbox=[x, y, x + 46, y + 88], confidence=round(self.confidence + 0.02, 3))
            )
        return self._stamp(detections, meta)


class NullDetector(BaseDetector):
    """不可用的检测器：真实模式下缺少 YOLO 依赖/权重时使用。

    绝不返回任何框——真实画面上宁可没有检测结果，也不能出现凭空生成的人员框。
    """

    backend = "unavailable"

    def detect(self, image: np.ndarray, meta: dict[str, Any] | None = None) -> list[Detection]:
        return []


class YoloDetector(BaseDetector):
    """预训练 YOLO 人员检测（ultralytics）。"""

    backend = "yolo"

    def __init__(
        self,
        model_path: Path,
        confidence: float = 0.1,
        device: str = "cpu",
        imgsz: int = 640,
        iou_threshold: float = 0.45,
        person_class: int = PERSON_CLASS_ID,
        max_det: int = 100,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:  # pragma: no cover - 依赖可选
            raise DetectorUnavailable(
                "未安装 ultralytics，无法加载 YOLO（pip install -r requirements-vision.txt）"
            ) from exc
        if not Path(model_path).exists():
            raise DetectorUnavailable(f"未找到 YOLO 权重文件: {model_path}")
        self.confidence = confidence
        self.device = device
        self.imgsz = imgsz
        self.iou_threshold = iou_threshold
        self.person_class = person_class
        self.max_det = max_det
        try:
            self._model = YOLO(str(model_path))
        except Exception as exc:  # noqa: BLE001
            raise DetectorUnavailable(f"加载 YOLO 模型失败: {exc}") from exc
        logger.info(
            "YOLO 模型已加载: %s (device=%s, imgsz=%d, conf=%.2f, iou=%.2f, max_det=%d)",
            model_path, device, imgsz, confidence, iou_threshold, max_det,
        )

    def detect(self, image: np.ndarray, meta: dict[str, Any] | None = None) -> list[Detection]:
        try:
            results = self._model.predict(
                image,
                conf=self.confidence,
                iou=self.iou_threshold,
                classes=[self.person_class],
                device=self.device,
                imgsz=self.imgsz,
                max_det=self.max_det,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - 推理失败不得终止采集线程
            logger.warning("YOLO 推理失败: %s", exc)
            return []
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls[0]) if box.cls is not None else self.person_class
                if cls != self.person_class:
                    continue
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                detections.append(Detection(bbox=xyxy, confidence=round(conf, 3)))
        return self._stamp(detections, meta)


def create_detector(
    *,
    simulation_mode: bool,
    model_path: Path,
    confidence: float,
    device: str,
    vision_enabled: bool,
    iou_threshold: float = 0.45,
    person_class: int = PERSON_CLASS_ID,
    max_det: int = 100,
    imgsz: int = 640,
) -> tuple[BaseDetector, str | None]:
    """返回 (detector, degraded_reason)。degraded_reason 非空表示未使用真实 YOLO。"""
    if not vision_enabled:
        detector: BaseDetector = NullDetector()
        detector.degraded_reason = "视觉推理已在配置中关闭"
        return detector, detector.degraded_reason
    if simulation_mode:
        # 只有模拟模式才允许模拟检测器（它的框来自模拟画面的真值）
        sim = SimulatedDetector(confidence)
        sim.degraded_reason = "模拟模式：使用模拟检测器"
        return sim, sim.degraded_reason
    try:
        return (
            YoloDetector(
                model_path,
                confidence,
                device,
                imgsz=imgsz,
                iou_threshold=iou_threshold,
                person_class=person_class,
                max_det=max_det,
            ),
            None,
        )
    except DetectorUnavailable as exc:
        # 真实模式下绝不退化成模拟检测器，否则真实画面会出现凭空生成的人员框
        logger.warning("YOLO 不可用，已停用人员检测: %s", exc)
        detector = NullDetector()
        detector.degraded_reason = str(exc)
        return detector, str(exc)


def benchmark(detector: BaseDetector, image: np.ndarray, rounds: int = 3) -> float:
    """返回平均推理耗时（毫秒）。"""
    start = time.perf_counter()
    for _ in range(rounds):
        detector.detect(image, {})
    return (time.perf_counter() - start) * 1000 / max(1, rounds)
