"""离线评测模块。

与在线系统解耦：评测只读取数据、跑推理、算指标，不依赖 FastAPI、
不碰数据库、不需要摄像头或串口，可以在任何一台机器上单独执行。
"""

from .mot import FrameData, MotAccumulator, MotMetrics, evaluate_sequence
from .mot_io import (
    Sequence,
    discover_sequences,
    iter_frames,
    read_mot_file,
    read_visibility,
    write_mot_file,
)
from .tracking import (
    SequenceResult,
    aggregate,
    evaluate_predictions,
    occlusion_breakdown,
    run_tracking,
)

__all__ = [
    "FrameData",
    "MotAccumulator",
    "MotMetrics",
    "evaluate_sequence",
    "Sequence",
    "discover_sequences",
    "iter_frames",
    "read_mot_file",
    "read_visibility",
    "write_mot_file",
    "SequenceResult",
    "aggregate",
    "evaluate_predictions",
    "occlusion_breakdown",
    "run_tracking",
]
