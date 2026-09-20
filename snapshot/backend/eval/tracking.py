"""跟踪评测流水线：图像序列 -> YOLO 检测 -> ByteTrack 跟踪 -> MOT 指标。

与在线系统共用同一套检测器与跟踪器实现（backend/vision/），
所以评测出来的数字就是系统真实跑出来的水平，不是另写一套"评测专用"代码。

遮挡分级统计：只有数据集官方提供 visibility 字段时才做，
字段缺失就直接跳过 —— 不自己编造遮挡等级。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Sequence as TypingSequence

from ..vision.detector import BaseDetector, Detection
from ..vision.tracker import ByteTracker
from .mot import FrameData, MotMetrics, evaluate_sequence

logger = logging.getLogger(__name__)

# 遮挡分级（依据数据集官方 visibility 字段，不是自己判断出来的）
OCCLUSION_LEVELS: list[tuple[str, float, float]] = [
    ("无遮挡", 0.8, 1.01),
    ("中度遮挡", 0.3, 0.8),
    ("严重遮挡", 0.0, 0.3),
]


@dataclass
class SequenceResult:
    """一段序列的评测结果。"""

    name: str
    metrics: MotMetrics
    predictions: dict[int, FrameData] = field(default_factory=dict)
    frames_processed: int = 0
    detections: int = 0
    inference_ms_total: float = 0.0
    tracking_ms_total: float = 0.0
    occlusion: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def inference_ms_per_frame(self) -> float:
        return self.inference_ms_total / self.frames_processed if self.frames_processed else 0.0

    @property
    def fps(self) -> float:
        total = self.inference_ms_total + self.tracking_ms_total
        return (self.frames_processed * 1000.0 / total) if total > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        data = self.metrics.as_dict()
        data.update(
            {
                "frames_processed": self.frames_processed,
                "detections": self.detections,
                "inference_ms_per_frame": round(self.inference_ms_per_frame, 3),
                "tracking_ms_per_frame": round(
                    self.tracking_ms_total / self.frames_processed
                    if self.frames_processed
                    else 0.0,
                    3,
                ),
                "fps": round(self.fps, 3),
            }
        )
        if self.occlusion:
            data["occlusion"] = self.occlusion
        return data


def run_tracking(
    frames: Iterable[tuple[int, Any]],
    detector: BaseDetector,
    tracker: ByteTracker,
    *,
    camera_id: str = "eval",
    progress: Callable[[int], None] | None = None,
) -> tuple[dict[int, FrameData], dict[str, float | int]]:
    """在一段序列上跑「检测 -> 跟踪」，返回逐帧预测与耗时统计。

    输出的框只来自检测器与跟踪器，不做任何后处理修饰。
    """
    predictions: dict[int, FrameData] = {}
    stats = {
        "frames": 0,
        "detections": 0,
        "inference_ms": 0.0,
        "tracking_ms": 0.0,
    }

    for frame_id, image in frames:
        started = time.perf_counter()
        detections: list[Detection] = detector.detect(
            image,
            {"camera_id": camera_id, "frame_id": frame_id, "timestamp": float(frame_id)},
        )
        detect_done = time.perf_counter()
        tracks = tracker.update(detections)
        track_done = time.perf_counter()

        stats["frames"] += 1
        stats["detections"] += len(detections)
        stats["inference_ms"] += (detect_done - started) * 1000.0
        stats["tracking_ms"] += (track_done - detect_done) * 1000.0

        predictions[frame_id] = FrameData(
            frame_id=frame_id,
            ids=[t.track_id for t in tracks],
            boxes=[[float(v) for v in t.bbox] for t in tracks],
        )
        if progress is not None:
            progress(frame_id)

    return predictions, stats


def evaluate_predictions(
    name: str,
    ground_truth: dict[int, FrameData],
    predictions: dict[int, FrameData],
    stats: dict[str, float | int] | None = None,
    *,
    iou_threshold: float = 0.5,
    visibility: dict[tuple[int, int], float] | None = None,
    eval_frames: set[int] | None = None,
) -> SequenceResult:
    """把预测与真值送进指标计算，并按需做遮挡分级统计。

    ``eval_frames`` 用于稀疏标注的数据集，限定只在官方标注帧上统计。
    """
    metrics = evaluate_sequence(
        ground_truth,
        predictions,
        iou_threshold=iou_threshold,
        sequence=name,
        eval_frames=eval_frames,
    )
    stats = stats or {}
    result = SequenceResult(
        name=name,
        metrics=metrics,
        predictions=predictions,
        frames_processed=int(stats.get("frames", len(predictions))),
        detections=int(stats.get("detections", 0)),
        inference_ms_total=float(stats.get("inference_ms", 0.0)),
        tracking_ms_total=float(stats.get("tracking_ms", 0.0)),
    )
    if visibility:
        result.occlusion = occlusion_breakdown(
            ground_truth, predictions, visibility, iou_threshold=iou_threshold
        )
    return result


def occlusion_breakdown(
    ground_truth: dict[int, FrameData],
    predictions: dict[int, FrameData],
    visibility: dict[tuple[int, int], float],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """按官方 visibility 字段分档统计召回率。

    只回答一个问题：目标被遮挡得越厉害，跟丢得越多吗？
    统计口径是"该档位下的真值目标有多少被跟到了"。
    """
    from .mot import iou

    buckets: dict[str, dict[str, Any]] = {
        label: {"gt": 0, "matched": 0, "recall": 0.0, "visibility_range": [low, high]}
        for label, low, high in OCCLUSION_LEVELS
    }

    for frame_id, gt_frame in ground_truth.items():
        pred_frame = predictions.get(frame_id)
        pred_boxes = pred_frame.boxes if pred_frame else []
        for track_id, gt_box in zip(gt_frame.ids, gt_frame.boxes):
            score = visibility.get((frame_id, track_id))
            if score is None:
                continue
            label = _bucket_for(score)
            if label is None:
                continue
            buckets[label]["gt"] += 1
            if any(iou(gt_box, p) >= iou_threshold for p in pred_boxes):
                buckets[label]["matched"] += 1

    for values in buckets.values():
        values["recall"] = (
            round(values["matched"] / values["gt"], 6) if values["gt"] else 0.0
        )
    return buckets


def _bucket_for(visibility: float) -> str | None:
    for label, low, high in OCCLUSION_LEVELS:
        if low <= visibility < high:
            return label
    return None


def aggregate(results: TypingSequence[SequenceResult]) -> dict[str, Any]:
    """把多段序列汇总成整体指标。

    汇总方式是**先把各序列的 TP/FP/FN/IDSW 加起来再算指标**，
    而不是对各序列的 MOTA 取平均 —— 后者会让短序列和长序列权重相同，
    是 MOT Challenge 明确不采用的做法。
    """
    if not results:
        return {"sequences": 0}

    gt = sum(r.metrics.gt_count for r in results)
    pred = sum(r.metrics.pred_count for r in results)
    tp = sum(r.metrics.true_positives for r in results)
    fp = sum(r.metrics.false_positives for r in results)
    fn = sum(r.metrics.false_negatives for r in results)
    idsw = sum(r.metrics.id_switches for r in results)
    idtp = sum(r.metrics.id_true_positives for r in results)
    idfp = sum(r.metrics.id_false_positives for r in results)
    idfn = sum(r.metrics.id_false_negatives for r in results)
    iou_sum = sum(r.metrics.matched_iou_sum for r in results)
    frames = sum(r.frames_processed for r in results)
    inference_ms = sum(r.inference_ms_total for r in results)
    tracking_ms = sum(r.tracking_ms_total for r in results)

    mota = 1.0 - (fn + fp + idsw) / gt if gt else 0.0
    idf1_den = 2 * idtp + idfp + idfn
    idf1 = 2 * idtp / idf1_den if idf1_den else 0.0

    return {
        "sequences": len(results),
        "frames": frames,
        "gt_count": gt,
        "pred_count": pred,
        "mota": round(mota, 6),
        "motp": round(iou_sum / tp, 6) if tp else 0.0,
        "idf1": round(idf1, 6),
        "precision": round(tp / (tp + fp), 6) if (tp + fp) else 0.0,
        "recall": round(tp / (tp + fn), 6) if (tp + fn) else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "id_switches": idsw,
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "mostly_tracked": sum(r.metrics.mostly_tracked for r in results),
        "partially_tracked": sum(r.metrics.partially_tracked for r in results),
        "mostly_lost": sum(r.metrics.mostly_lost for r in results),
        "inference_ms_per_frame": round(inference_ms / frames, 3) if frames else 0.0,
        "tracking_ms_per_frame": round(tracking_ms / frames, 3) if frames else 0.0,
        "fps": round(frames * 1000.0 / (inference_ms + tracking_ms), 3)
        if (inference_ms + tracking_ms) > 0
        else 0.0,
    }
