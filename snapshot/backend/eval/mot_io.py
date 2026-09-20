"""MOT Challenge 格式的读写与序列发现。

MOT Challenge（以及 PersonPath22 等沿用该格式的数据集）的标注是每行一条记录的
CSV，字段依次为：

    frame, id, x, y, w, h, conf, class, visibility

约定：
- frame 与 id 从 1 开始；
- x, y 是**左上角**坐标，w, h 是宽高（本项目内部统一用 [x1, y1, x2, y2]，
  读取时立刻转换，避免两套坐标约定在代码里混着走）；
- gt.txt 中 conf 为 0 表示该条标注被忽略（distractor / 忽略区域），
  **必须剔除**，否则会把不该算的目标算成漏检，压低 MOTA；
- class 为 1 才是行人（MOT 的类别定义），其它类别（车辆、遮挡物等）同样剔除；
- visibility 是可见比例，用于遮挡分级统计。

标准目录结构：

    <dataset_root>/<sequence_name>/
        img1/000001.jpg ...        图像序列
        gt/gt.txt                  真值标注
        seqinfo.ini                序列信息（可选）
"""

from __future__ import annotations

import configparser
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .mot import FrameData

# MOT Challenge 中行人的类别号
PEDESTRIAN_CLASS = 1

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class MotAnnotation:
    """一条标注记录。"""

    frame_id: int
    track_id: int
    bbox: list[float]  # [x1, y1, x2, y2]
    confidence: float = 1.0
    class_id: int = PEDESTRIAN_CLASS
    visibility: float = 1.0


@dataclass
class Sequence:
    """一段可评测的视频序列。"""

    name: str
    root: Path
    image_dir: Path | None = None
    gt_file: Path | None = None
    video_file: Path | None = None
    width: int = 0
    height: int = 0
    frame_count: int = 0
    fps: float = 0.0

    @property
    def has_ground_truth(self) -> bool:
        return self.gt_file is not None and self.gt_file.exists()

    @property
    def annotated_frames_file(self) -> Path | None:
        if self.gt_file is None:
            return None
        candidate = self.gt_file.parent / "annotated_frames.txt"
        return candidate if candidate.exists() else None

    def images(self) -> list[Path]:
        if self.image_dir is None or not self.image_dir.exists():
            return []
        return sorted(
            p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )


def parse_mot_line(row: Iterable[str]) -> MotAnnotation | None:
    """解析一行标注。字段不足或不是数字则返回 None（跳过该行，不中断整份文件）。"""
    values = [v.strip() for v in row if v.strip() != ""]
    if len(values) < 6:
        return None
    try:
        frame_id = int(float(values[0]))
        track_id = int(float(values[1]))
        x = float(values[2])
        y = float(values[3])
        w = float(values[4])
        h = float(values[5])
    except ValueError:
        return None

    confidence = _optional_float(values, 6, 1.0)
    class_id = int(_optional_float(values, 7, PEDESTRIAN_CLASS))
    visibility = _optional_float(values, 8, 1.0)

    return MotAnnotation(
        frame_id=frame_id,
        track_id=track_id,
        bbox=[x, y, x + w, y + h],
        confidence=confidence,
        class_id=class_id,
        visibility=visibility,
    )


def _optional_float(values: list[str], index: int, default: float) -> float:
    if index >= len(values):
        return default
    try:
        return float(values[index])
    except ValueError:
        return default


def read_mot_file(
    path: Path,
    *,
    keep_classes: set[int] | None = None,
    min_confidence: float = 0.0,
    min_visibility: float = 0.0,
    drop_zero_confidence: bool = True,
) -> dict[int, FrameData]:
    """读取一份 MOT 格式标注，按帧组织。

    默认只保留行人类别、且 conf > 0 的条目 —— 这是 MOT Challenge 官方评测的做法，
    把被标记为忽略的目标算进来会人为压低指标。
    """
    # keep_classes=None -> 只保留行人；传空集合 -> 不做类别过滤（读预测文件时用）
    keep = keep_classes if keep_classes is not None else {PEDESTRIAN_CLASS}
    frames: dict[int, FrameData] = {}

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            annotation = parse_mot_line(row)
            if annotation is None:
                continue
            if drop_zero_confidence and annotation.confidence <= 0:
                continue
            if annotation.confidence < min_confidence:
                continue
            if keep and annotation.class_id not in keep:
                continue
            # MOT 预测格式用 -1 表示"该字段未提供"，不是"可见度为 0"。
            # 把未提供当成 0 会把整份预测文件过滤成空。
            if annotation.visibility >= 0 and annotation.visibility < min_visibility:
                continue
            frame = frames.setdefault(
                annotation.frame_id, FrameData(frame_id=annotation.frame_id)
            )
            frame.ids.append(annotation.track_id)
            frame.boxes.append(list(annotation.bbox))
    return frames


def read_visibility(path: Path) -> dict[tuple[int, int], float]:
    """读取 (帧号, 轨迹号) -> 可见比例，用于遮挡分级统计。

    只有数据集**官方提供** visibility 字段时才有意义；字段缺失时返回空表，
    上层据此跳过遮挡分析 —— 绝不自己编造遮挡等级。
    """
    table: dict[tuple[int, int], float] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            annotation = parse_mot_line(row)
            if annotation is None or len(row) < 9:
                continue
            # 负值表示该字段未提供（MOT 预测文件常写 -1），不是真实的可见比例
            if annotation.visibility < 0:
                continue
            table[(annotation.frame_id, annotation.track_id)] = annotation.visibility
    return table


def read_annotated_frames(path: Path) -> set[int] | None:
    """读取"官方标注了哪些帧"的清单。

    稀疏标注的数据集（如 PersonPath22 每 5 帧标 1 帧）必须只在标注帧上评测，
    否则未标注帧上的预测会被全部算成误检，指标失真。
    文件不存在时返回 None，表示"逐帧标注，全部帧都参与评测"。
    """
    path = Path(path)
    if not path.exists():
        return None
    frames: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            frames.add(int(float(line)))
        except ValueError:
            continue
    return frames or None


def write_mot_file(path: Path, frames: dict[int, FrameData]) -> int:
    """把预测结果写成 MOT 格式，便于用官方工具二次核对。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for frame_id in sorted(frames):
            frame = frames[frame_id]
            for track_id, bbox in zip(frame.ids, frame.boxes):
                x1, y1, x2, y2 = bbox
                writer.writerow(
                    [
                        frame_id,
                        track_id,
                        round(x1, 2),
                        round(y1, 2),
                        round(x2 - x1, 2),
                        round(y2 - y1, 2),
                        1,
                        -1,
                        -1,
                        -1,
                    ]
                )
                written += 1
    return written


def read_seqinfo(path: Path) -> dict[str, str]:
    """读取 seqinfo.ini。文件缺失或损坏都只返回空表，不抛异常。"""
    parser = configparser.ConfigParser()
    try:
        parser.read(path, encoding="utf-8")
    except (configparser.Error, OSError):
        return {}
    if not parser.has_section("Sequence"):
        return {}
    return dict(parser["Sequence"])


def discover_sequences(root: Path) -> list[Sequence]:
    """在数据集根目录下发现所有序列。

    识别两种布局：标准 MOT 目录（img1/ + gt/gt.txt）与"视频文件 + 同名标注"。
    """
    root = Path(root)
    if not root.exists():
        return []

    sequences: list[Sequence] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        image_dir = child / "img1"
        gt_file = child / "gt" / "gt.txt"
        video = next(
            (v for v in sorted(child.glob("*.mp4")) if v.is_file()), None
        )
        if not image_dir.exists() and video is None:
            continue

        info = read_seqinfo(child / "seqinfo.ini")
        sequence = Sequence(
            name=child.name,
            root=child,
            image_dir=image_dir if image_dir.exists() else None,
            gt_file=gt_file if gt_file.exists() else None,
            video_file=video,
            width=int(info.get("imwidth", 0) or 0),
            height=int(info.get("imheight", 0) or 0),
            frame_count=int(info.get("seqlength", 0) or 0),
            fps=float(info.get("framerate", 0) or 0),
        )
        if not sequence.frame_count:
            sequence.frame_count = len(sequence.images())
        sequences.append(sequence)
    return sequences


def iter_frames(sequence: Sequence, limit: int = 0) -> Iterator[tuple[int, "object"]]:
    """按帧号顺序产出 (帧号, 图像)。帧号从 1 开始，与 MOT 标注对齐。

    图像序列优先；没有图像序列时才去解码视频文件。
    """
    import cv2  # 局部导入：读标注、算指标都不需要 OpenCV

    images = sequence.images()
    if images:
        for index, image_path in enumerate(images, start=1):
            if limit and index > limit:
                return
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            yield index, image
        return

    if sequence.video_file is None:
        return

    capture = cv2.VideoCapture(str(sequence.video_file))
    try:
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                return
            index += 1
            if limit and index > limit:
                return
            yield index, image
    finally:
        capture.release()
