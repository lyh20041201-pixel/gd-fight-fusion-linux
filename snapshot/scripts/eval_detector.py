"""检测器评测：在公开数据集上评估预训练 YOLO 的 person 类别性能，并实测本机 FPS。

本项目只使用官方预训练权重，不做自训练；本脚本用于回答"这个预训练模型
在本任务上到底有多准、在本机能跑多快"，为毕业设计报告提供量化依据。

用法：
    .venv\\Scripts\\python.exe scripts\\eval_detector.py                    # 快速冒烟（coco128，自动下载 ~7MB）
    .venv\\Scripts\\python.exe scripts\\eval_detector.py --data coco.yaml   # 完整评测（COCO val2017，5000 张，需下载 ~1GB）
    .venv\\Scripts\\python.exe scripts\\eval_detector.py --fps-only         # 只测速度

输出：
    docs/thesis/评测结果-检测器.md   （报告可直接引用的表格）
    runs/                            （ultralytics 生成的 PR 曲线、混淆矩阵等图）
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "thesis" / "评测结果-检测器.md"

PERSON_CLASS_ID = 0


def _fmt(x: float, n: int = 4) -> str:
    return f"{x:.{n}f}"


def measure_fps(
    model, imgsz: int, width: int, height: int, rounds: int, warmup: int,
    device: str = "cpu",
) -> dict:
    """在与系统实际采集分辨率一致的合成帧上实测推理速度。"""
    import numpy as np
    import torch

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    on_gpu = device not in {"cpu", ""}

    for _ in range(warmup):
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)
    if on_gpu:
        torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)
        # GPU 是异步执行的，不同步的话测到的只是下发命令的时间，不是真实耗时
        if on_gpu:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.fmean(samples)
    return {
        "resolution": f"{width}x{height}",
        "device": device,
        "imgsz": imgsz,
        "rounds": rounds,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[max(0, int(len(samples) * 0.95) - 1)],
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }


def evaluate(model, data: str, imgsz: int, device: str = "cpu") -> dict | None:
    """在给定数据集上评测，提取 person 类别的指标。"""
    metrics = model.val(data=data, imgsz=imgsz, device=device, verbose=False, plots=True)
    box = metrics.box

    idx = None
    for i, cls in enumerate(list(box.ap_class_index)):
        if int(cls) == PERSON_CLASS_ID:
            idx = i
            break
    if idx is None:
        print(f"警告：数据集 {data} 的评测结果中没有 person 类别，跳过按类指标。")
        return None

    p, r, ap50, ap = box.class_result(idx)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "data": data,
        "device": device,
        "imgsz": imgsz,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "map50": float(ap50),
        "map50_95": float(ap),
        "all_map50": float(box.map50),
        "all_map50_95": float(box.map),
        "speed": dict(metrics.speed),
        "save_dir": str(getattr(metrics, "save_dir", "")),
    }


def write_report(weights: str, det: dict | None, fps: dict, env: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# 检测器评测结果\n")
    lines.append("> 本文件由 `scripts/eval_detector.py` 自动生成，请勿手工编辑。\n")
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("\n## 1. 评测环境\n")
    lines.append("| 项 | 值 |")
    lines.append("| --- | --- |")
    for k, v in env.items():
        lines.append(f"| {k} | {v} |")
    lines.append(f"| 权重 | {weights} |")

    if det:
        lines.append("\n## 2. person 类别检测精度\n")
        lines.append(f"数据集：`{det['data']}`，推理尺寸 {det['imgsz']}×{det['imgsz']}\n")
        lines.append("| 指标 | 数值 |")
        lines.append("| --- | --- |")
        lines.append(f"| 查准率 Precision | {_fmt(det['precision'])} |")
        lines.append(f"| 查全率 Recall | {_fmt(det['recall'])} |")
        lines.append(f"| F1 值 | {_fmt(det['f1'])} |")
        lines.append(f"| mAP@0.5 | {_fmt(det['map50'])} |")
        lines.append(f"| mAP@0.5:0.95 | {_fmt(det['map50_95'])} |")
        lines.append("\n全类别参考值（非本系统关注指标）："
                     f"mAP@0.5 = {_fmt(det['all_map50'])}，"
                     f"mAP@0.5:0.95 = {_fmt(det['all_map50_95'])}\n")
        sp = det["speed"]
        lines.append("评测阶段单帧耗时（ms）：" + "，".join(f"{k} {v:.2f}" for k, v in sp.items()))
        if det.get("save_dir"):
            lines.append(f"\nPR 曲线与混淆矩阵等图表位于：`{det['save_dir']}`")

    lines.append("\n## 3. 本机推理速度\n")
    lines.append(f"输入分辨率 {fps['resolution']}（与系统摄像头采集配置一致），"
                 f"推理尺寸 {fps['imgsz']}，{fps['rounds']} 次取统计值\n")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 平均单帧耗时 | {fps['mean_ms']:.1f} ms |")
    lines.append(f"| 中位数 | {fps['median_ms']:.1f} ms |")
    lines.append(f"| P95 | {fps['p95_ms']:.1f} ms |")
    lines.append(f"| 平均帧率 | **{fps['fps']:.2f} FPS** |")
    lines.append(f"\n按此速度，单进程串行处理 N 路摄像头时每路可达约 {fps['fps']:.2f}/N FPS；"
                 "系统默认采集 15 FPS，超出部分由有界帧缓冲丢弃旧帧。\n")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入: {OUT}")


def write_json(out_dir: Path, weights: str, det: dict | None, fps: dict, env: dict) -> None:
    """机器可读指标。报告里的数字必须能追溯到这份文件，而不是手工敲上去的。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weights": weights,
        "environment": env,
        "detection": det,
        "speed": fps,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    import csv

    rows = []
    if det:
        rows.append({
            "dataset": det["data"], "device": det["device"], "imgsz": det["imgsz"],
            "precision": det["precision"], "recall": det["recall"], "f1": det["f1"],
            "map50": det["map50"], "map50_95": det["map50_95"],
            "mean_ms": fps["mean_ms"], "fps": fps["fps"],
        })
    else:
        rows.append({
            "dataset": "", "device": fps["device"], "imgsz": fps["imgsz"],
            "mean_ms": fps["mean_ms"], "fps": fps["fps"],
        })
    with (out_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--data", default="coco128.yaml",
                    help="ultralytics 数据集配置，如 coco128.yaml（快速）或 coco.yaml（完整 val2017）")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--width", type=int, default=640, help="FPS 实测输入宽度，与 CAMERA_WIDTH 一致")
    ap.add_argument("--height", type=int, default=480, help="FPS 实测输入高度，与 CAMERA_HEIGHT 一致")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--fps-only", action="store_true")
    ap.add_argument(
        "--device", default="cpu",
        help="推理设备：cpu 或 0（第一块 GPU）。报告中必须写明用的是哪个。",
    )
    ap.add_argument(
        "--json-out", default=str(ROOT / "results" / "detection"),
        help="机器可读指标的输出目录",
    )
    args = ap.parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"缺少依赖：{exc}\n请先执行： pip install -r requirements-vision.txt")
        return 1

    weights = Path(args.weights)
    if not weights.exists():
        print(f"权重不存在：{weights}\n请先执行： python scripts/download_yolo.py yolov8n.pt")
        return 1

    env = {
        "操作系统": f"{platform.system()} {platform.release()}",
        "处理器": platform.processor() or "未知",
        "Python": platform.python_version(),
        "PyTorch": f"{torch.__version__}（CUDA 可用：{torch.cuda.is_available()}）",
        "推理设备": args.device.upper() if args.device == "cpu" else f"GPU (cuda:{args.device})",
    }
    if args.device != "cpu" and torch.cuda.is_available():
        env["GPU"] = torch.cuda.get_device_name(0)
    print("评测环境：" + json.dumps(env, ensure_ascii=False, indent=2))

    model = YOLO(str(weights))

    det = None
    if not args.fps_only:
        print(f"\n[1/2] 在 {args.data} 上评测 person 类别 ...")
        det = evaluate(model, args.data, args.imgsz, args.device)

    print(f"\n[{'1/1' if args.fps_only else '2/2'}] 实测本机推理速度 ...")
    fps = measure_fps(
        model, args.imgsz, args.width, args.height, args.rounds, args.warmup, args.device
    )
    print(f"  平均 {fps['mean_ms']:.1f} ms/帧 → {fps['fps']:.2f} FPS")

    write_report(str(weights.relative_to(ROOT)), det, fps, env)
    write_json(Path(args.json_out), str(weights.relative_to(ROOT)), det, fps, env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
