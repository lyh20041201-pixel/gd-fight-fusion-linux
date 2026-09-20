"""跟踪器评测：在 MOT 格式数据集上评估 YOLO + ByteTrack 的 MOTA / IDF1 / IDSW。

对应开题报告第三章"（4）模型评价算法 · 跟踪器评价"。
评测用的检测器与跟踪器就是系统在线运行时用的那一套（backend/vision/），
所以跑出来的数字代表系统的真实水平。

用法：
    # 完整评测
    .venv\\Scripts\\python.exe scripts\\eval_tracking.py --data datasets/personpath22/tracking/test

    # 冒烟测试：只取每段序列的前 50 帧，先确认流水线通了
    .venv\\Scripts\\python.exe scripts\\eval_tracking.py --data <路径> --limit-frames 50

    # 对照实验：贪心分配 vs 匈牙利分配
    .venv\\Scripts\\python.exe scripts\\eval_tracking.py --data <路径> --matcher greedy --tag greedy

数据集目录结构（MOT Challenge 标准）：
    <data>/<序列名>/img1/000001.jpg ...
    <data>/<序列名>/gt/gt.txt
    <data>/<序列名>/seqinfo.ini   （可选）

输出：
    results/tracking/<tag>/summary.json      汇总指标（机器可读）
    results/tracking/<tag>/summary.csv       汇总指标（可直接贴进报告）
    results/tracking/<tag>/per_sequence.csv  逐序列指标
    results/tracking/<tag>/config.json       本次评测的完整参数与环境
    results/tracking/<tag>/predictions/*.txt 预测结果（MOT 格式，可用官方工具复核）
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.vision import load_vision_config  # noqa: E402
from backend.eval.mot_io import (  # noqa: E402
    discover_sequences,
    iter_frames,
    read_annotated_frames,
    read_mot_file,
    read_visibility,
    write_mot_file,
)
from backend.eval.tracking import (  # noqa: E402
    aggregate,
    evaluate_predictions,
    run_tracking,
)
from backend.vision.detector import DetectorUnavailable, YoloDetector  # noqa: E402
from backend.vision.tracker import ByteTracker  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    config = load_vision_config(ROOT)
    parser = argparse.ArgumentParser(description="YOLO + ByteTrack 跟踪评测")
    parser.add_argument("--data", required=True, help="MOT 格式数据集根目录")
    parser.add_argument("--weights", default=str(ROOT / config.detector.model_path))
    parser.add_argument("--device", default=config.detector.device, help="cpu / cuda")
    parser.add_argument("--imgsz", type=int, default=config.detector.imgsz)
    parser.add_argument("--conf", type=float, default=config.detector.conf_threshold)
    parser.add_argument("--iou", type=float, default=config.detector.iou_threshold)
    parser.add_argument("--max-det", type=int, default=config.detector.max_det)
    parser.add_argument(
        "--matcher",
        default=config.tracker.matcher,
        choices=["hungarian", "greedy"],
        help="数据关联算法，用于做对照实验",
    )
    parser.add_argument(
        "--eval-iou", type=float, default=0.5, help="评测时判定命中的 IoU 阈值"
    )
    parser.add_argument(
        "--limit-frames", type=int, default=0, help="每段序列只取前 N 帧（0 表示不限制）"
    )
    parser.add_argument("--sequences", nargs="*", default=None, help="只评测指定序列")
    parser.add_argument("--tag", default="", help="结果子目录名，默认按数据集目录名")
    parser.add_argument(
        "--output", default=str(ROOT / "results" / "tracking"), help="结果输出根目录"
    )
    return parser


def environment(args) -> dict:
    info = {
        "操作系统": f"{platform.system()} {platform.release()}",
        "处理器": platform.processor() or "未知",
        "Python": platform.python_version(),
        "推理设备": args.device,
    }
    try:
        import torch

        info["PyTorch"] = torch.__version__
        info["CUDA 可用"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["GPU"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["PyTorch"] = "未安装"
    try:
        import ultralytics

        info["Ultralytics"] = ultralytics.__version__
    except ImportError:
        info["Ultralytics"] = "未安装"
    return info


def main() -> int:
    args = build_parser().parse_args()

    data_root = Path(args.data)
    if not data_root.exists():
        print(f"数据集目录不存在：{data_root}")
        print("请先按 datasets/README.md 下载并转换数据集。")
        return 1

    sequences = discover_sequences(data_root)
    if args.sequences:
        wanted = set(args.sequences)
        sequences = [s for s in sequences if s.name in wanted]
    if not sequences:
        print(f"在 {data_root} 下没有发现任何序列。")
        print("期望的目录结构：<数据集>/<序列名>/img1/*.jpg 与 <序列名>/gt/gt.txt")
        return 1

    without_gt = [s.name for s in sequences if not s.has_ground_truth]
    if without_gt:
        print(f"以下序列缺少 gt/gt.txt，将跳过：{', '.join(without_gt)}")
        sequences = [s for s in sequences if s.has_ground_truth]
    if not sequences:
        print("没有任何带真值标注的序列，无法评测。")
        return 1

    weights = Path(args.weights)
    try:
        detector = YoloDetector(
            weights,
            confidence=args.conf,
            device=args.device,
            imgsz=args.imgsz,
            iou_threshold=args.iou,
            max_det=args.max_det,
        )
    except DetectorUnavailable as exc:
        print(f"无法加载检测器：{exc}")
        return 1

    tag = args.tag or data_root.name
    out_dir = Path(args.output) / tag
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    config = load_vision_config(ROOT)
    started_at = time.time()
    results = []

    for index, sequence in enumerate(sequences, start=1):
        print(f"[{index}/{len(sequences)}] 评测序列 {sequence.name} ...", flush=True)
        ground_truth = read_mot_file(sequence.gt_file)
        if not ground_truth:
            print(f"  跳过：{sequence.gt_file} 中没有有效的行人标注")
            continue

        # 稀疏标注的数据集（PersonPath22 每 5 帧标 1 帧）只在官方标注帧上评测，
        # 否则未标注帧上的预测会被全部算成误检。清单由转换脚本从原始标注写出。
        eval_frames = None
        if sequence.annotated_frames_file is not None:
            eval_frames = read_annotated_frames(sequence.annotated_frames_file)
            if eval_frames:
                print(f"  按官方标注帧评测：{len(eval_frames)} 帧")

        if args.limit_frames:
            ground_truth = {
                f: data for f, data in ground_truth.items() if f <= args.limit_frames
            }
            if eval_frames:
                eval_frames = {f for f in eval_frames if f <= args.limit_frames}

        tracker = ByteTracker(
            track_thresh=config.tracker.track_thresh,
            low_thresh=config.tracker.low_thresh,
            match_thresh=config.tracker.match_thresh,
            second_match_thresh=config.tracker.second_match_thresh,
            track_buffer=config.tracker.track_buffer,
            min_hits=config.tracker.min_hits,
            matcher=args.matcher,
            camera_id=sequence.name,
        )

        predictions, stats = run_tracking(
            iter_frames(sequence, limit=args.limit_frames),
            detector,
            tracker,
            camera_id=sequence.name,
        )
        if not predictions:
            print(f"  跳过：{sequence.name} 没能读出任何图像帧")
            continue

        # 遮挡分级只在数据集官方提供 visibility 时才做
        visibility = read_visibility(sequence.gt_file)
        result = evaluate_predictions(
            sequence.name,
            ground_truth,
            predictions,
            stats,
            iou_threshold=args.eval_iou,
            visibility=visibility or None,
            eval_frames=eval_frames,
        )
        results.append(result)

        write_mot_file(out_dir / "predictions" / f"{sequence.name}.txt", predictions)
        m = result.metrics
        print(
            f"  MOTA {m.mota:.4f} · IDF1 {m.idf1:.4f} · IDSW {m.id_switches} · "
            f"FP {m.false_positives} · FN {m.false_negatives} · "
            f"{result.frames_processed} 帧 · {result.fps:.1f} FPS"
        )

    if not results:
        print("没有任何序列评测成功。")
        return 1

    summary = aggregate(results)
    summary["matcher"] = args.matcher
    summary["eval_iou"] = args.eval_iou
    summary["elapsed_seconds"] = round(time.time() - started_at, 1)

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "per_sequence": [r.as_dict() for r in results],
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_csv(out_dir / "summary.csv", [summary])
    _write_csv(
        out_dir / "per_sequence.csv",
        [{k: v for k, v in r.as_dict().items() if k != "occlusion"} for r in results],
    )

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "data": str(data_root),
                "weights": str(weights),
                "arguments": vars(args),
                "vision_config": config.as_dict(),
                "environment": environment(args),
                "sequences": [s.name for s in sequences],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n===== 汇总 =====")
    print(f"序列数 {summary['sequences']} · 帧数 {summary['frames']}")
    print(f"MOTA  {summary['mota']:.4f}")
    print(f"IDF1  {summary['idf1']:.4f}")
    print(f"IDSW  {summary['id_switches']}")
    print(f"FP {summary['fp']} · FN {summary['fn']}")
    print(f"结果已写入：{out_dir}")
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    sys.exit(main())
