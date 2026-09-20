"""下载预训练 YOLO 权重到 models/ 目录（真实模式使用）。

用法：
    .venv\\Scripts\\python.exe scripts\\download_yolo.py [yolov8n.pt]

说明：
- 本项目只使用官方预训练权重的 person 类别，不做自训练；
- 需要先安装可选依赖：pip install -r requirements-vision.txt
- 没有权重文件时系统会自动降级为模拟检测器并把 YOLO 模块标记为降级状态。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "yolov8n.pt"
    MODELS.mkdir(parents=True, exist_ok=True)
    target = MODELS / name
    if target.exists():
        print(f"权重已存在: {target}")
        return 0

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print("未安装 ultralytics，请先执行： pip install -r requirements-vision.txt")
        return 1

    print(f"正在下载 {name} ...")
    model = YOLO(name)  # ultralytics 会自动下载到缓存目录
    source = Path(getattr(model, "ckpt_path", "") or name)
    if source.exists() and source.resolve() != target.resolve():
        shutil.copy(source, target)
    if target.exists():
        print(f"完成: {target}")
        print("请在 .env 中确认 YOLO_MODEL_PATH=models/" + name)
        return 0
    print(f"下载后未找到文件，请手动把 {name} 放到 {MODELS}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
