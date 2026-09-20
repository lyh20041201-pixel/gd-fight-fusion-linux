"""ROI 区域判定。

人员是否属于某个区域，依据目标框"底部中心点"（人的落脚点）是否落在
归一化多边形 ROI 内。ROI 使用 0~1 归一化坐标，与分辨率无关。
"""

from __future__ import annotations

from typing import Sequence

Point = tuple[float, float]
Polygon = Sequence[Sequence[float]]

FULL_FRAME_ROI: list[list[float]] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """射线法判定点是否在多边形内（含边界近似）。"""
    if not polygon or len(polygon) < 3:
        return True
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def bottom_center_normalized(bbox: Sequence[float], width: int, height: int) -> Point:
    x1, _y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    cx = (float(x1) + float(x2)) / 2.0
    return (cx / max(1, width), float(y2) / max(1, height))


def in_roi(bbox: Sequence[float], roi: Polygon, width: int, height: int) -> bool:
    return point_in_polygon(bottom_center_normalized(bbox, width, height), roi)


def default_roi_for_index(index: int) -> list[list[float]]:
    """为模拟摄像头生成互不相同的默认 ROI，便于演示区域划分。"""
    presets = [
        [[0.04, 0.34], [0.96, 0.34], [0.96, 0.98], [0.04, 0.98]],
        [[0.08, 0.40], [0.92, 0.40], [0.98, 0.98], [0.02, 0.98]],
        [[0.02, 0.45], [0.98, 0.45], [0.98, 0.99], [0.02, 0.99]],
        [[0.10, 0.38], [0.90, 0.38], [0.94, 0.96], [0.06, 0.96]],
    ]
    return presets[index % len(presets)]
