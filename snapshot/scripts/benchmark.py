"""系统性能基准：把"这套系统在这台机器上到底能跑多快"量化下来。

测什么：
1. YOLO 单帧推理延迟与 FPS（CPU 与 GPU 分别测）
2. ByteTrack 单帧关联耗时（按目标数量分档）
3. 规则引擎单次评估耗时
4. WebSocket 广播序列化耗时
5. 多路视频源压力测试（1 / 2 / 4 路）

关于第 5 项的措辞：如果没有 4 台真实摄像头，就用录制视频或模拟画面作为输入源。
这种情况下报告里**必须写"视频源压力测试"，不能写"四台实际摄像头测试"**。
本脚本会在结果里明确标注 source_kind，避免日后写报告时记混。

用法：
    .venv\\Scripts\\python.exe scripts\\benchmark.py                 # 全部
    .venv\\Scripts\\python.exe scripts\\benchmark.py --device 0      # 用 GPU 测视觉部分
    .venv\\Scripts\\python.exe scripts\\benchmark.py --skip-vision   # 只测非视觉部分

输出：results/benchmark/{metrics.json, summary.md}
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def _stats(samples: list[float]) -> dict:
    if not samples:
        return {}
    ordered = sorted(samples)
    mean = statistics.fmean(ordered)
    return {
        "rounds": len(ordered),
        "mean_ms": round(mean, 4),
        "median_ms": round(statistics.median(ordered), 4),
        "p95_ms": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 4),
        "max_ms": round(ordered[-1], 4),
        "ops_per_second": round(1000.0 / mean, 2) if mean > 0 else 0.0,
    }


# ---------------- 1. YOLO ----------------


def bench_detector(device: str, imgsz: int, width: int, height: int, rounds: int) -> dict:
    from backend.vision.detector import DetectorUnavailable, YoloDetector

    weights = ROOT / "models" / "yolov8n.pt"
    try:
        detector = YoloDetector(weights, confidence=0.35, device=device, imgsz=imgsz)
    except DetectorUnavailable as exc:
        return {"error": str(exc)}

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)

    on_gpu = device not in {"cpu", ""}
    for _ in range(5):
        detector.detect(frame)
    if on_gpu:
        import torch

        torch.cuda.synchronize()

    samples = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        detector.detect(frame)
        if on_gpu:
            import torch

            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)

    result = _stats(samples)
    result.update({"device": device, "imgsz": imgsz, "resolution": f"{width}x{height}"})
    result["fps"] = result.pop("ops_per_second", 0.0)
    return result


# ---------------- 2. ByteTrack ----------------


def bench_tracker(rounds: int = 500) -> dict:
    from backend.vision.detector import Detection
    from backend.vision.tracker import ByteTracker

    results = {}
    for target_count in (2, 5, 10, 30, 60):
        tracker = ByteTracker(min_hits=1)
        rng = np.random.default_rng(1)
        # 先建立轨迹
        base = [
            [float(x), 50.0, float(x) + 40, 190.0]
            for x in rng.integers(0, 560, target_count)
        ]
        for _ in range(3):
            tracker.update([Detection(bbox=list(b), confidence=0.9) for b in base])

        samples = []
        for _ in range(rounds):
            jitter = [
                [v + float(rng.normal(0, 1.5)) for v in box] for box in base
            ]
            detections = [Detection(bbox=b, confidence=0.9) for b in jitter]
            t0 = time.perf_counter()
            tracker.update(detections)
            samples.append((time.perf_counter() - t0) * 1000.0)
        results[f"{target_count}_targets"] = _stats(samples)
    return results


def bench_matcher_comparison(rounds: int = 300) -> dict:
    """匈牙利 vs 贪心的耗时对比（论文里要说明为什么用得起全局最优）。"""
    from backend.vision.assignment import greedy, hungarian

    results = {}
    for size in (5, 10, 20, 50):
        rng = np.random.default_rng(2)
        matrices = [
            [[float(v) for v in row] for row in rng.random((size, size))]
            for _ in range(10)
        ]
        for name, solver in (("hungarian", hungarian), ("greedy", greedy)):
            samples = []
            for _ in range(rounds // 10):
                for matrix in matrices:
                    t0 = time.perf_counter()
                    solver(matrix)
                    samples.append((time.perf_counter() - t0) * 1000.0)
            results[f"{name}_{size}x{size}"] = _stats(samples)
    return results


# ---------------- 3. 规则引擎 ----------------


def bench_rule_engine(rounds: int = 2000) -> dict:
    from backend.config.defaults import DEFAULT_RULES, DEFAULT_SCHEDULE, DEFAULT_THRESHOLDS
    from backend.rules.engine import CameraSnapshot, ClassroomSnapshot, RuleEngine

    class _Runtime:
        """最小运行时配置桩：基准测试不该依赖数据库。"""

        thresholds = dict(DEFAULT_THRESHOLDS)
        schedule = dict(DEFAULT_SCHEDULE)

        def threshold(self, key, default=None):
            return self.thresholds.get(key, default)

    engine = RuleEngine(_Runtime(), DEFAULT_RULES)
    cameras = [
        CameraSnapshot(f"CAM-{i:02d}", True, False, 8, f"区域{i}", [f"ID-{j}" for j in range(8)])
        for i in range(1, 5)
    ]

    samples = []
    for i in range(rounds):
        snapshot = ClassroomSnapshot(
            timestamp=time.time() + i * 0.5,
            sensors={"co2": 800.0, "noise": 55.0, "smoke": 0.0, "temperature": 24.0,
                     "humidity": 50.0, "light": 300.0},
            sensor_online={"co2": True, "noise": True, "smoke": True},
            occupancy_total=32,
            occupancy_raw=32,
            cameras=cameras,
        )
        t0 = time.perf_counter()
        engine.evaluate(snapshot)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return _stats(samples)


# ---------------- 4. WebSocket 广播 ----------------


def bench_broadcast(rounds: int = 2000) -> dict:
    """测的是消息序列化开销（网络发送不由本进程决定）。"""
    from backend.schemas.common import OccupancyUpdate

    samples = []
    for i in range(rounds):
        update = OccupancyUpdate(
            total=30 + (i % 5),
            per_camera={f"CAM-{c:02d}": 7 + (i % 3) for c in range(1, 5)},
            timestamp=time.time(),
            trend="stable",
        )
        t0 = time.perf_counter()
        payload = update.model_dump(mode="json")
        json.dumps(payload)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return _stats(samples)


# ---------------- 5. 多路视频源压力测试 ----------------


def bench_multi_source(device: str, counts: tuple[int, ...], seconds: float) -> dict:
    """用模拟摄像头作为视频源，测 1/2/4 路并发时每路能跑到多少 FPS。

    **这是视频源压力测试，不是"四台实际摄像头测试"。**
    结果里的 source_kind 字段会明确写出来。
    """
    import threading

    from backend.cameras.sim_source import SimulatedCamera
    from backend.vision.detector import DetectorUnavailable, YoloDetector
    from backend.vision.tracker import ByteTracker

    weights = ROOT / "models" / "yolov8n.pt"
    try:
        detector = YoloDetector(weights, confidence=0.35, device=device, imgsz=640)
    except DetectorUnavailable as exc:
        return {"error": str(exc)}

    results: dict = {"source_kind": "simulated_video_source", "device": device}
    lock = threading.Lock()

    for count in counts:
        cameras = [
            SimulatedCamera(f"CAM-{i:02d}", f"压测{i}", index=i, width=640, height=480,
                            fps=30, base_people=3)
            for i in range(count)
        ]
        for camera in cameras:
            camera.open()
        trackers = [ByteTracker() for _ in range(count)]
        frame_counts = [0] * count
        stop = threading.Event()

        def worker(index: int) -> None:
            while not stop.is_set():
                frame = cameras[index].read()
                if frame is None:
                    continue
                # 与线上一致：所有摄像头共用一个检测器，靠锁串行化推理
                with lock:
                    detections = detector.detect(
                        frame.image,
                        {"camera_id": f"CAM-{index:02d}", "frame_id": frame_counts[index]},
                    )
                trackers[index].update(detections)
                frame_counts[index] += 1

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(count)]
        started = time.perf_counter()
        for thread in threads:
            thread.start()
        time.sleep(seconds)
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
        elapsed = time.perf_counter() - started

        for camera in cameras:
            camera.close()

        total = sum(frame_counts)
        results[f"{count}_sources"] = {
            "cameras": count,
            "seconds": round(elapsed, 2),
            "frames_total": total,
            "fps_total": round(total / elapsed, 2),
            "fps_per_camera": round(total / elapsed / count, 2),
            "per_camera_frames": frame_counts,
        }
    return results


# ---------------- 汇总 ----------------


def environment(device: str) -> dict:
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "未知",
        "python": platform.python_version(),
        "device": device,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = "未安装"
    return info


def render_markdown(report: dict) -> str:
    lines = ["# 系统性能基准\n"]
    lines.append("> 本文件由 `scripts/benchmark.py` 自动生成，请勿手工编辑。\n")
    lines.append(f"生成时间：{report['generated_at']}\n")

    lines.append("\n## 运行环境\n")
    lines.append("| 项 | 值 |")
    lines.append("| --- | --- |")
    for key, value in report["environment"].items():
        lines.append(f"| {key} | {value} |")

    detector = report.get("detector", {})
    if detector and "error" not in detector:
        lines.append("\n## 1. YOLO 单帧推理\n")
        lines.append("| 设备 | 分辨率 | 平均 | 中位 | P95 | FPS |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for device, item in detector.items():
            if "error" in item:
                continue
            lines.append(
                f"| {device} | {item['resolution']} | {item['mean_ms']:.2f} ms | "
                f"{item['median_ms']:.2f} ms | {item['p95_ms']:.2f} ms | "
                f"**{item['fps']:.1f}** |"
            )

    tracker = report.get("tracker", {})
    if tracker:
        lines.append("\n## 2. ByteTrack 单帧关联\n")
        lines.append("| 目标数 | 平均 | P95 |")
        lines.append("| --- | --- | --- |")
        for key, item in tracker.items():
            lines.append(
                f"| {key.replace('_targets', '')} | {item['mean_ms']:.4f} ms | "
                f"{item['p95_ms']:.4f} ms |"
            )

    matcher = report.get("matcher", {})
    if matcher:
        lines.append("\n## 3. 分配算法耗时（匈牙利 vs 贪心）\n")
        lines.append("| 矩阵规模 | 匈牙利 | 贪心 |")
        lines.append("| --- | --- | --- |")
        sizes = sorted({k.split("_")[1] for k in matcher})
        for size in sizes:
            h = matcher.get(f"hungarian_{size}", {})
            g = matcher.get(f"greedy_{size}", {})
            lines.append(
                f"| {size} | {h.get('mean_ms', 0):.4f} ms | {g.get('mean_ms', 0):.4f} ms |"
            )
        lines.append(
            "\n教室场景每路目标数在数十以内，匈牙利算法的额外开销可以忽略，"
            "因此采用全局最优的匈牙利匹配（与开题报告一致）。\n"
        )

    for key, title in (("rules", "4. 规则引擎单次评估"), ("broadcast", "5. WebSocket 消息序列化")):
        item = report.get(key)
        if item:
            lines.append(f"\n## {title}\n")
            lines.append("| 指标 | 数值 |")
            lines.append("| --- | --- |")
            lines.append(f"| 平均 | {item['mean_ms']:.4f} ms |")
            lines.append(f"| 中位 | {item['median_ms']:.4f} ms |")
            lines.append(f"| P95 | {item['p95_ms']:.4f} ms |")
            lines.append(f"| 每秒可处理 | {item['ops_per_second']:.0f} 次 |")

    multi = report.get("multi_source", {})
    if multi and "error" not in multi:
        lines.append("\n## 6. 多路视频源压力测试\n")
        lines.append(
            f"> 输入源类型：`{multi.get('source_kind')}`。"
            "**这是视频源压力测试，不是四台实际摄像头的现场测试。**\n"
        )
        lines.append("| 路数 | 总 FPS | 每路 FPS | 是否满足 15 FPS 采集 |")
        lines.append("| --- | --- | --- | --- |")
        for key in sorted(k for k in multi if k.endswith("_sources")):
            item = multi[key]
            ok = "是" if item["fps_per_camera"] >= 15 else "否（由有界缓冲丢弃旧帧）"
            lines.append(
                f"| {item['cameras']} | {item['fps_total']:.1f} | "
                f"{item['fps_per_camera']:.1f} | {ok} |"
            )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="系统性能基准")
    parser.add_argument("--device", default="", help="视觉部分用的设备；留空则 CPU 与 GPU 都测")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--stress-seconds", type=float, default=8.0)
    parser.add_argument("--skip-vision", action="store_true", help="跳过需要模型的项")
    parser.add_argument("--skip-stress", action="store_true", help="跳过多路压力测试")
    parser.add_argument("--out", default=str(ROOT / "results" / "benchmark"))
    args = parser.parse_args()

    report: dict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": environment(args.device or "auto"),
    }

    devices: list[str] = []
    if not args.skip_vision:
        if args.device:
            devices = [args.device]
        else:
            devices = ["cpu"]
            try:
                import torch

                if torch.cuda.is_available():
                    devices.append("0")
            except ImportError:
                pass

    if devices:
        print("[1/6] YOLO 单帧推理 ...")
        report["detector"] = {
            ("GPU" if d not in {"cpu", ""} else "CPU"): bench_detector(
                d, 640, 640, 480, args.rounds
            )
            for d in devices
        }
        for name, item in report["detector"].items():
            if "error" in item:
                print(f"  {name}: {item['error']}")
            else:
                print(f"  {name}: {item['mean_ms']:.2f} ms -> {item['fps']:.1f} FPS")

    print("[2/6] ByteTrack 关联 ...")
    report["tracker"] = bench_tracker()
    print("[3/6] 分配算法对比 ...")
    report["matcher"] = bench_matcher_comparison()
    print("[4/6] 规则引擎 ...")
    report["rules"] = bench_rule_engine()
    print(f"  平均 {report['rules']['mean_ms']:.4f} ms")
    print("[5/6] WebSocket 序列化 ...")
    report["broadcast"] = bench_broadcast()

    if devices and not args.skip_stress:
        print("[6/6] 多路视频源压力测试 ...")
        report["multi_source"] = bench_multi_source(
            devices[-1], (1, 2, 4), args.stress_seconds
        )
        for key in sorted(k for k in report["multi_source"] if k.endswith("_sources")):
            item = report["multi_source"][key]
            print(f"  {item['cameras']} 路: 每路 {item['fps_per_camera']:.1f} FPS")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"\n结果已写入：{out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
