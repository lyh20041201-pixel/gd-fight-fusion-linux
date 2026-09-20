"""视觉流水线参数的集中配置。

为什么单独一份 config/vision.yaml：
检测阈值、NMS IoU、跟踪门限这些是**算法参数**，改动会直接影响检测与跟踪指标，
必须能和实验结果一起归档、一起写进毕业设计报告；
它们既不该散落在代码里（改一次要翻好几个文件），
也不适合混进部署用的 .env（.env 管的是端口、串口、密钥这类环境相关项）。

加载优先级：config/vision.yaml > .env > 代码默认值。
文件不存在、格式损坏、缺字段都不会导致启动失败 —— 一律回落到默认值并记日志。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_FILE = "config/vision.yaml"


@dataclass
class DetectorConfig:
    model_path: str = "models/yolov8n.pt"
    device: str = "cpu"
    imgsz: int = 640
    conf_threshold: float = 0.1
    iou_threshold: float = 0.45
    person_class: int = 0
    max_det: int = 100


@dataclass
class TrackerConfig:
    track_thresh: float = 0.5
    low_thresh: float = 0.1
    match_thresh: float = 0.75
    second_match_thresh: float = 0.5
    track_buffer: int = 30
    min_hits: int = 2
    matcher: str = "hungarian"


@dataclass
class ObstructionConfig:
    variance: float = 40.0
    hold_seconds: float = 3.0


@dataclass
class VisionConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    obstruction: ObstructionConfig = field(default_factory=ObstructionConfig)
    source: str = "defaults"  # defaults / vision.yaml，用于在实验记录里说明参数来源

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce(section: Any, target: Any, name: str) -> None:
    """把 YAML 里读到的键写进 dataclass，类型不对就跳过并记日志。"""
    if not isinstance(section, dict):
        return
    for key, value in section.items():
        if not hasattr(target, key):
            logger.warning("%s 中存在未知配置项 %s.%s，已忽略", CONFIG_FILE, name, key)
            continue
        expected = type(getattr(target, key))
        try:
            setattr(target, key, expected(value))
        except (TypeError, ValueError):
            logger.warning(
                "%s 中 %s.%s 的值 %r 类型不合法，保留默认值", CONFIG_FILE, name, key, value
            )


def load_vision_config(base_dir: Path, path: str = CONFIG_FILE) -> VisionConfig:
    """读取视觉参数。任何异常都回落到默认值，绝不让配置问题拖垮启动。"""
    config = VisionConfig()
    file_path = base_dir / path
    if not file_path.exists():
        logger.info("未找到 %s，视觉参数使用代码默认值", path)
        return config

    try:
        import yaml  # type: ignore
    except ImportError:
        logger.warning("未安装 PyYAML，无法读取 %s，视觉参数使用代码默认值", path)
        return config

    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - 配置损坏不得导致启动失败
        logger.error("读取 %s 失败（%s），视觉参数使用代码默认值", path, exc)
        return config

    if not isinstance(raw, dict):
        logger.error("%s 内容不是映射结构，视觉参数使用代码默认值", path)
        return config

    _coerce(raw.get("detector"), config.detector, "detector")
    _coerce(raw.get("tracker"), config.tracker, "tracker")
    _coerce(raw.get("obstruction"), config.obstruction, "obstruction")
    config.source = path
    logger.info("已加载视觉参数: %s", path)
    return config
