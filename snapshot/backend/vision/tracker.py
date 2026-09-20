"""ByteTrack 人员跟踪（纯 Python 实现，无额外依赖）。

遵循 ByteTrack 的核心思想：
1) 先用高分检测框与现有轨迹做一次关联；
2) 再用低分检测框与仍未匹配的轨迹做第二次关联，找回被遮挡目标；
3) 未匹配的高分检测创建新轨迹，长期未命中的轨迹进入 lost 并最终移除。

匹配采用 IoU 代价 + **匈牙利算法（Kuhn-Munkres）** 求全局最优分配，
与开题报告第三章的描述一致；也可通过 matcher="greedy" 切换为贪心分配做对照实验。
轨迹在丢失期间按常速度模型外推位置（见 Track.predict）。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterable

from .assignment import INFEASIBLE, solve
from .detector import Detection

logger = logging.getLogger(__name__)


def iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    """一条匿名人员轨迹。

    track_id 只是**单摄像头、短时间内**的匿名编号，不是任何人的身份标识，
    也不跨摄像头复用 —— 系统不做人脸识别，不记录身份信息。
    """

    track_id: int
    bbox: list[float]
    score: float
    state: str = "tracked"  # tracked / lost
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0])
    history: list[list[float]] = field(default_factory=list)
    camera_id: str = ""
    start_time: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    @property
    def duration(self) -> float:
        """该轨迹已持续的时长（秒），用于"长时间停留"一类判定。"""
        return max(0.0, self.last_seen - self.start_time)

    @property
    def label(self) -> str:
        return f"ID-{self.track_id}"

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    def predict(self) -> list[float]:
        vx, vy = self.velocity
        x1, y1, x2, y2 = self.bbox
        return [x1 + vx, y1 + vy, x2 + vx, y2 + vy]

    def update(self, bbox: list[float], score: float, timestamp: float | None = None) -> None:
        old_cx = (self.bbox[0] + self.bbox[2]) / 2
        old_cy = (self.bbox[1] + self.bbox[3]) / 2
        new_cx = (bbox[0] + bbox[2]) / 2
        new_cy = (bbox[1] + bbox[3]) / 2
        self.velocity = [
            0.6 * self.velocity[0] + 0.4 * (new_cx - old_cx),
            0.6 * self.velocity[1] + 0.4 * (new_cy - old_cy),
        ]
        self.bbox = list(bbox)
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        self.state = "tracked"
        self.last_seen = timestamp if timestamp is not None else time.time()
        self.history.append([new_cx, new_cy])
        if len(self.history) > 60:
            self.history = self.history[-60:]

    def mark_lost(self) -> None:
        self.state = "lost"
        self.time_since_update += 1
        # 丢失期间按上次速度外推，便于短时遮挡后找回
        self.bbox = self.predict()


class ByteTracker:
    """ByteTrack 主体。每路摄像头一个独立实例。"""

    def __init__(
        self,
        track_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.75,
        second_match_thresh: float = 0.5,
        track_buffer: int = 30,
        min_hits: int = 2,
        matcher: str = "hungarian",
        camera_id: str = "",
    ) -> None:
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh
        self.track_buffer = track_buffer
        self.min_hits = min_hits
        # hungarian = 全局最优（开题报告采用）；greedy = 局部最优，仅用于对照实验
        self.matcher = matcher
        self.camera_id = camera_id
        self._tracks: list[Track] = []
        self._next_id = 1
        self.frame_count = 0

    # ---------- 主流程 ----------
    def update(self, detections: Iterable[Detection]) -> list[Track]:
        self.frame_count += 1
        dets = list(detections)
        high = [d for d in dets if d.confidence >= self.track_thresh]
        low = [d for d in dets if self.low_thresh <= d.confidence < self.track_thresh]

        for track in self._tracks:
            track.age += 1

        unmatched_tracks = list(self._tracks)

        # 第一阶段：高分检测
        matches, unmatched_tracks, unmatched_high = self._associate(
            unmatched_tracks, high, self.match_thresh
        )
        for track, det in matches:
            track.update(det.bbox, det.confidence, det.timestamp or None)

        # 第二阶段：低分检测找回被遮挡目标
        matches2, unmatched_tracks, _unmatched_low = self._associate(
            unmatched_tracks, low, self.second_match_thresh
        )
        for track, det in matches2:
            track.update(det.bbox, det.confidence, det.timestamp or None)

        for track in unmatched_tracks:
            track.mark_lost()

        for det in unmatched_high:
            started = det.timestamp or time.time()
            self._tracks.append(
                Track(
                    track_id=self._next_id,
                    bbox=list(det.bbox),
                    score=det.confidence,
                    camera_id=det.camera_id or self.camera_id,
                    start_time=started,
                    last_seen=started,
                )
            )
            self._next_id += 1

        # 清理长期丢失的轨迹
        self._tracks = [
            t for t in self._tracks if t.time_since_update <= self.track_buffer
        ]
        return self.active_tracks()

    def _associate(
        self, tracks: list[Track], dets: list[Detection], threshold: float
    ) -> tuple[list[tuple[Track, Detection]], list[Track], list[Detection]]:
        """一次关联：构造 IoU 代价矩阵，求最小代价分配。

        代价用 1 - IoU；代价高于门限的配对置为不可行（inf），
        保证"宁可不匹配，也不做明显错误的关联"。
        """
        if not tracks or not dets:
            return [], tracks, dets

        cost: list[list[float]] = []
        for track in tracks:
            # 丢失中的轨迹先按常速度模型外推，再与检测框计算重合度
            predicted = track.predict() if track.time_since_update > 0 else track.bbox
            row = []
            for det in dets:
                score = iou(predicted, det.bbox)
                distance = 1.0 - score
                row.append(distance if distance <= threshold else INFEASIBLE)
            cost.append(row)

        assignment = solve(cost, self.matcher)

        used_tracks = {ti for ti, _dj in assignment}
        used_dets = {dj for _ti, dj in assignment}
        matches = [(tracks[ti], dets[dj]) for ti, dj in assignment]
        unmatched_tracks = [t for i, t in enumerate(tracks) if i not in used_tracks]
        unmatched_dets = [d for i, d in enumerate(dets) if i not in used_dets]
        return matches, unmatched_tracks, unmatched_dets

    # ---------- 查询 ----------
    def active_tracks(self) -> list[Track]:
        return [
            t
            for t in self._tracks
            if t.state == "tracked" and (t.hits >= self.min_hits or self.frame_count <= 2)
        ]

    @property
    def all_tracks(self) -> list[Track]:
        return list(self._tracks)

    def summary(self) -> dict[str, object]:
        active = self.active_tracks()
        return {
            "active": len(active),
            "total_created": self._next_id - 1,
            "lost": sum(1 for t in self._tracks if t.state == "lost"),
            "frames": self.frame_count,
            "track_ids": [t.label for t in active],
        }

    def reset(self) -> None:
        self._tracks.clear()
        self.frame_count = 0
