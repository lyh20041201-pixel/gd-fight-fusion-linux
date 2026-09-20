"""多目标跟踪评测（CLEAR MOT + IDF1）。

对应开题报告第三章"（4）模型评价算法 · 跟踪器评价"：

    MOTA = 1 − (FN + FP + IDSW) / GT
    IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)

实现严格按 MOT Challenge 的标准做法，没有自创任何公式：

- 逐帧关联：**先保留上一帧已经成立的匹配**（只要重合度仍然达标），
  剩下的再用匈牙利算法求最小代价分配。这一条"沿用历史匹配"的规则出自
  Bernardin & Stiefelhagen 的 CLEAR MOT 定义，直接决定 IDSW 的计数结果，
  漏掉它算出来的 IDSW 会明显偏大。
- IDSW：某个真值目标本帧匹配到的预测编号，与它上一次匹配到的编号不同时 +1。
- IDF1：**在整段序列上**做一次真值编号与预测编号的全局最优一一对应
  （最大化两者同时出现且重合达标的帧数 IDTP），而不是逐帧统计。

正确性验证方式见 tests/test_mot_metrics.py：
既有人工构造、可以手算的小序列，也与成熟库 py-motmetrics 做交叉验证。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..vision.assignment import INFEASIBLE, hungarian

Box = Sequence[float]  # [x1, y1, x2, y2]

DEFAULT_IOU_THRESHOLD = 0.5


def iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a[0], box_a[1], box_a[2], box_a[3]
    bx1, by1, bx2, by2 = box_b[0], box_b[1], box_b[2], box_b[3]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class MotMetrics:
    """一段序列的跟踪评测结果。"""

    frames: int = 0
    gt_count: int = 0            # 真值目标出现总次数（GT）
    pred_count: int = 0          # 预测目标输出总次数
    true_positives: int = 0
    false_positives: int = 0     # FP：预测出来但没有对应真值
    false_negatives: int = 0     # FN：真值存在但没被跟到（漏检）
    id_switches: int = 0         # IDSW：身份切换次数
    matched_iou_sum: float = 0.0
    id_true_positives: int = 0   # IDTP
    id_false_positives: int = 0  # IDFP
    id_false_negatives: int = 0  # IDFN
    mostly_tracked: int = 0      # 被跟踪覆盖 ≥80% 的真值轨迹数
    partially_tracked: int = 0
    mostly_lost: int = 0         # 被跟踪覆盖 <20% 的真值轨迹数
    gt_tracks: int = 0
    pred_tracks: int = 0
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    sequence: str = ""

    @property
    def mota(self) -> float:
        """MOTA = 1 − (FN + FP + IDSW) / GT。可能为负（错误数超过真值数）。"""
        if self.gt_count == 0:
            return 0.0
        return 1.0 - (
            self.false_negatives + self.false_positives + self.id_switches
        ) / self.gt_count

    @property
    def motp(self) -> float:
        """匹配上的目标的平均重合度。越接近 1 表示框越准。"""
        if self.true_positives == 0:
            return 0.0
        return self.matched_iou_sum / self.true_positives

    @property
    def idf1(self) -> float:
        """IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)。"""
        denominator = 2 * self.id_true_positives + self.id_false_positives + self.id_false_negatives
        if denominator == 0:
            return 0.0
        return 2 * self.id_true_positives / denominator

    @property
    def idp(self) -> float:
        denominator = self.id_true_positives + self.id_false_positives
        return self.id_true_positives / denominator if denominator else 0.0

    @property
    def idr(self) -> float:
        denominator = self.id_true_positives + self.id_false_negatives
        return self.id_true_positives / denominator if denominator else 0.0

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    def as_dict(self) -> dict[str, Any]:
        """机器可读的指标字典，直接写进 results/ 供报告引用。"""
        return {
            "sequence": self.sequence,
            "iou_threshold": self.iou_threshold,
            "frames": self.frames,
            "gt_count": self.gt_count,
            "pred_count": self.pred_count,
            "gt_tracks": self.gt_tracks,
            "pred_tracks": self.pred_tracks,
            "mota": round(self.mota, 6),
            "motp": round(self.motp, 6),
            "idf1": round(self.idf1, 6),
            "idp": round(self.idp, 6),
            "idr": round(self.idr, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "id_switches": self.id_switches,
            "idtp": self.id_true_positives,
            "idfp": self.id_false_positives,
            "idfn": self.id_false_negatives,
            "mostly_tracked": self.mostly_tracked,
            "partially_tracked": self.partially_tracked,
            "mostly_lost": self.mostly_lost,
        }


class MotAccumulator:
    """逐帧累计跟踪评测统计量。

    用法：对序列里的每一帧调用一次 ``update``，最后 ``result()`` 取指标。
    """

    def __init__(
        self,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        sequence: str = "",
    ) -> None:
        self.iou_threshold = iou_threshold
        self.sequence = sequence

        self._frames = 0
        self._gt_count = 0
        self._pred_count = 0
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._idsw = 0
        self._iou_sum = 0.0

        # gt_id -> 上一次匹配到的 pred_id，用于沿用历史匹配与统计 IDSW
        self._last_match: dict[Any, Any] = {}
        # (gt_id, pred_id) -> 同时出现且重合达标的帧数，用于 IDF1 的全局匹配
        self._pair_frames: dict[tuple[Any, Any], int] = {}
        self._gt_frames: dict[Any, int] = {}
        self._pred_frames: dict[Any, int] = {}
        self._gt_matched_frames: dict[Any, int] = {}

    def update(
        self,
        gt_ids: Sequence[Any],
        gt_boxes: Sequence[Box],
        pred_ids: Sequence[Any],
        pred_boxes: Sequence[Box],
    ) -> dict[str, Any]:
        """处理一帧，返回该帧的统计（便于逐帧调试）。"""
        if len(gt_ids) != len(gt_boxes) or len(pred_ids) != len(pred_boxes):
            raise ValueError("编号与边界框数量不一致")

        self._frames += 1
        self._gt_count += len(gt_ids)
        self._pred_count += len(pred_ids)
        for gid in gt_ids:
            self._gt_frames[gid] = self._gt_frames.get(gid, 0) + 1
        for pid in pred_ids:
            self._pred_frames[pid] = self._pred_frames.get(pid, 0) + 1

        overlaps = [[iou(g, p) for p in pred_boxes] for g in gt_boxes]

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        matches: list[tuple[int, int]] = []

        # 第一步：沿用上一帧已成立的匹配（CLEAR MOT 的核心规则）。
        # 少了这一步，目标稍有抖动就会被重新分配，IDSW 会被严重高估。
        pred_index = {pid: i for i, pid in enumerate(pred_ids)}
        for gi, gid in enumerate(gt_ids):
            previous = self._last_match.get(gid)
            if previous is None:
                continue
            pi = pred_index.get(previous)
            if pi is None or pi in matched_pred:
                continue
            if overlaps[gi][pi] >= self.iou_threshold:
                matched_gt.add(gi)
                matched_pred.add(pi)
                matches.append((gi, pi))

        # 第二步：剩下的用匈牙利算法求最小代价分配
        free_gt = [i for i in range(len(gt_ids)) if i not in matched_gt]
        free_pred = [j for j in range(len(pred_ids)) if j not in matched_pred]
        if free_gt and free_pred:
            cost = [
                [
                    (1.0 - overlaps[gi][pj])
                    if overlaps[gi][pj] >= self.iou_threshold
                    else INFEASIBLE
                    for pj in free_pred
                ]
                for gi in free_gt
            ]
            for row, col in hungarian(cost):
                matches.append((free_gt[row], free_pred[col]))
                matched_gt.add(free_gt[row])
                matched_pred.add(free_pred[col])

        # 统计
        frame_idsw = 0
        for gi, pj in matches:
            gid, pid = gt_ids[gi], pred_ids[pj]
            self._tp += 1
            self._iou_sum += overlaps[gi][pj]
            self._pair_frames[(gid, pid)] = self._pair_frames.get((gid, pid), 0) + 1
            self._gt_matched_frames[gid] = self._gt_matched_frames.get(gid, 0) + 1
            previous = self._last_match.get(gid)
            if previous is not None and previous != pid:
                self._idsw += 1
                frame_idsw += 1
            self._last_match[gid] = pid

        frame_fn = len(gt_ids) - len(matches)
        frame_fp = len(pred_ids) - len(matches)
        self._fn += frame_fn
        self._fp += frame_fp

        return {
            "frame": self._frames,
            "matches": len(matches),
            "fp": frame_fp,
            "fn": frame_fn,
            "idsw": frame_idsw,
        }

    def result(self) -> MotMetrics:
        idtp, idfp, idfn = self._identity_matching()
        mt, pt, ml = self._track_coverage()
        return MotMetrics(
            frames=self._frames,
            gt_count=self._gt_count,
            pred_count=self._pred_count,
            true_positives=self._tp,
            false_positives=self._fp,
            false_negatives=self._fn,
            id_switches=self._idsw,
            matched_iou_sum=self._iou_sum,
            id_true_positives=idtp,
            id_false_positives=idfp,
            id_false_negatives=idfn,
            mostly_tracked=mt,
            partially_tracked=pt,
            mostly_lost=ml,
            gt_tracks=len(self._gt_frames),
            pred_tracks=len(self._pred_frames),
            iou_threshold=self.iou_threshold,
            sequence=self.sequence,
        )

    # ---------- IDF1 的全局身份匹配 ----------
    def _identity_matching(self) -> tuple[int, int, int]:
        """在整段序列上给真值编号与预测编号做一次最优一一对应。

        目标是最大化 IDTP（两者同时出现且重合达标的帧数）。匈牙利算法求的是
        最小代价，所以这里取负值。
        """
        gt_ids = sorted(self._gt_frames, key=str)
        pred_ids = sorted(self._pred_frames, key=str)
        idtp = 0
        if gt_ids and pred_ids:
            cost = [
                [-float(self._pair_frames.get((gid, pid), 0)) for pid in pred_ids]
                for gid in gt_ids
            ]
            for row, col in hungarian(cost):
                idtp += self._pair_frames.get((gt_ids[row], pred_ids[col]), 0)
        idfn = self._gt_count - idtp
        idfp = self._pred_count - idtp
        return idtp, idfp, idfn

    # ---------- MT / PT / ML ----------
    def _track_coverage(self) -> tuple[int, int, int]:
        """按 MOT Challenge 定义统计真值轨迹被跟踪覆盖的程度。"""
        mostly_tracked = partially_tracked = mostly_lost = 0
        for gid, total in self._gt_frames.items():
            if total == 0:
                continue
            ratio = self._gt_matched_frames.get(gid, 0) / total
            if ratio >= 0.8:
                mostly_tracked += 1
            elif ratio < 0.2:
                mostly_lost += 1
            else:
                partially_tracked += 1
        return mostly_tracked, partially_tracked, mostly_lost


@dataclass
class FrameData:
    """一帧的真值或预测。"""

    frame_id: int
    ids: list[Any] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)


def evaluate_sequence(
    ground_truth: dict[int, FrameData] | Iterable[FrameData],
    predictions: dict[int, FrameData] | Iterable[FrameData],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    sequence: str = "",
    eval_frames: set[int] | None = None,
) -> MotMetrics:
    """评测一整段序列。

    默认情况下，真值帧与预测帧的并集都参与评测：预测里多出来的帧同样计入 FP，
    不能靠"只评测有真值的帧"把误检藏起来。

    ``eval_frames`` 用于**稀疏标注**的数据集（例如 PersonPath22 每 5 帧才标 1 帧）：
    传入官方标注帧清单后，只在这些帧上统计。此时未标注帧上的预测既不算命中
    也不算误检 —— 因为那些帧根本没有真值可比。这个清单必须来自数据集本身，
    不能用"预测了哪些帧"反推，否则等于把误检藏起来。
    """
    gt_map = _as_map(ground_truth)
    pred_map = _as_map(predictions)

    frame_ids = set(gt_map) | set(pred_map)
    if eval_frames is not None:
        frame_ids &= eval_frames

    accumulator = MotAccumulator(iou_threshold=iou_threshold, sequence=sequence)
    for frame_id in sorted(frame_ids):
        gt = gt_map.get(frame_id, FrameData(frame_id))
        pred = pred_map.get(frame_id, FrameData(frame_id))
        accumulator.update(gt.ids, gt.boxes, pred.ids, pred.boxes)
    return accumulator.result()


def _as_map(data: dict[int, FrameData] | Iterable[FrameData]) -> dict[int, FrameData]:
    if isinstance(data, dict):
        return data
    return {frame.frame_id: frame for frame in data}
