"""跨域对比评测：把每个模型都放到**全部**测试集上评一遍。

为什么必须跨域评：
只在各自的测试集上评测，每个微调模型看起来都会"变好"——因为它就是在那个域上训的。
真正要回答的问题是**微调有没有牺牲别的能力**：
在课堂数据上微调之后，通用人员检测掉了多少？遮挡场景掉了多少？
不做跨域评测，"域适配提升了 X%"这句话就是自欺欺人。

评测矩阵：

              | COCO val2017 | SCB test | PersonPath22 test
    Model A   |      ✔       |    ✔     |        ✔
    Model B   |      ✔       |    ✔     |        ✔
    Model C   |      ✔       |    ✔     |        ✔
    Model D   |      ✔       |    ✔     |        ✔

用法：
    .venv\\Scripts\\python.exe scripts\\compare_models.py --device 0
    .venv\\Scripts\\python.exe scripts\\compare_models.py --models baseline scb
    .venv\\Scripts\\python.exe scripts\\compare_models.py --skip-coco   # COCO 较慢，可跳过

输出：
    results/comparison/matrix.json   完整结果
    results/comparison/matrix.csv    可直接贴进报告
    results/comparison/summary.md    带解读的表格
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.eval.experiment import environment_report  # noqa: E402
from tools.datasets.common import read_split  # noqa: E402

SPLIT_DIR = ROOT / "datasets" / "splits"
PERSON_CLASS_ID = 0

# 待评模型：实验名 -> 权重路径（相对项目根）
MODELS = {
    "baseline": "models/yolov8n.pt",
    "scb": "experiments/runs/scb/train/weights/best.pt",
    "personpath": "experiments/runs/personpath/train/weights/best.pt",
    "mixed": "experiments/runs/mixed/train/weights/best.pt",
}

MODEL_LABELS = {
    "baseline": "Model A：COCO 预训练基线",
    "scb": "Model B：+SCB 课堂微调",
    "personpath": "Model C：+PersonPath22 遮挡微调",
    "mixed": "Model D：联合微调",
}

# 待评数据集：名称 -> (数据集根, 划分前缀)
DATASETS = {
    "scb": (ROOT / "datasets" / "scb" / "yolo", "scb"),
    "personpath22": (ROOT / "datasets" / "personpath22", "personpath22"),
}


def build_test_yaml(name: str, root: Path, prefix: str, out_dir: Path) -> Path | None:
    """为某个数据集的**测试集**生成一份 ultralytics data.yaml。"""
    import yaml

    entries = read_split(SPLIT_DIR / f"{prefix}_test.txt")
    if not entries:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    list_file = out_dir / f"{name}_test.txt"
    list_file.write_text(
        "\n".join(str((root / e).resolve()) for e in entries) + "\n", encoding="utf-8"
    )
    data_yaml = out_dir / f"{name}_test.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": str(list_file.resolve()),
                "val": str(list_file.resolve()),
                "names": {PERSON_CLASS_ID: "person"},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def extract(metrics, person_only: bool) -> dict:
    """从 ultralytics 结果里取指标。COCO 是多类别的，要单独抽 person。"""
    box = metrics.box
    out: dict = {}
    if person_only:
        index = None
        for i, cls in enumerate(list(box.ap_class_index)):
            if int(cls) == PERSON_CLASS_ID:
                index = i
                break
        if index is None:
            return {"error": "结果中没有 person 类别"}
        p, r, ap50, ap = box.class_result(index)
    else:
        p, r, ap50, ap = float(box.mp), float(box.mr), float(box.map50), float(box.map)

    p, r = float(p), float(r)
    out.update(
        {
            "precision": round(p, 6),
            "recall": round(r, 6),
            "f1": round(2 * p * r / (p + r), 6) if (p + r) > 0 else 0.0,
            "map50": round(float(ap50), 6),
            "map50_95": round(float(ap), 6),
        }
    )
    speed = getattr(metrics, "speed", None)
    if speed:
        out["inference_ms"] = round(float(speed.get("inference", 0.0)), 3)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="跨域对比评测")
    parser.add_argument("--models", nargs="*", default=None, help="只评指定模型")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001, help="评测用低阈值，让 PR 曲线完整")
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--skip-coco", action="store_true", help="跳过 COCO（5000 张较慢）")
    parser.add_argument("--out", default=str(ROOT / "results" / "comparison"))
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[FAIL] 未安装 ultralytics")
        return 1

    wanted = args.models or list(MODELS)
    available: dict[str, Path] = {}
    for name in wanted:
        if name not in MODELS:
            print(f"[!] 未知模型 {name}，跳过")
            continue
        path = ROOT / MODELS[name]
        if path.exists():
            available[name] = path
        else:
            print(f"[!] {name} 的权重尚不存在（{MODELS[name]}），跳过 —— 该实验还没训练")

    if not available:
        print("[FAIL] 没有任何可评测的模型。请先运行 train.py。")
        return 1

    out_dir = Path(args.out)
    work_dir = out_dir / "_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 组装评测目标
    targets: list[tuple[str, str, bool]] = []  # (数据集名, data.yaml, person_only)
    if not args.skip_coco:
        targets.append(("COCO val2017", "coco.yaml", True))
    for name, (root, prefix) in DATASETS.items():
        if not root.exists():
            print(f"[!] 数据集 {name} 不存在，跳过")
            continue
        data_yaml = build_test_yaml(name, root, prefix, work_dir)
        if data_yaml is None:
            print(f"[!] 数据集 {name} 没有测试集划分清单，跳过")
            continue
        targets.append((name, str(data_yaml), False))

    if not targets:
        print("[FAIL] 没有任何可用的测试集。")
        return 1

    print(f"模型 {len(available)} 个 × 测试集 {len(targets)} 个 = {len(available) * len(targets)} 次评测\n")

    results: dict[str, dict] = {}
    started = time.time()
    for model_name, weights in available.items():
        results[model_name] = {"weights": str(weights.relative_to(ROOT)), "datasets": {}}
        model = YOLO(str(weights))
        for dataset_name, data, person_only in targets:
            print(f"[{model_name}] 在 {dataset_name} 上评测 ...", flush=True)
            try:
                metrics = model.val(
                    data=data,
                    split="val",
                    imgsz=args.imgsz,
                    device=args.device,
                    conf=args.conf,
                    iou=args.iou,
                    plots=False,
                    verbose=False,
                )
                entry = extract(metrics, person_only)
            except Exception as exc:  # noqa: BLE001 - 单项失败不影响整张矩阵
                entry = {"error": f"{type(exc).__name__}: {exc}"}
            results[model_name]["datasets"][dataset_name] = entry
            if "error" in entry:
                print(f"    失败：{entry['error']}")
            else:
                print(
                    f"    mAP50 {entry['map50']:.4f}  mAP50-95 {entry['map50_95']:.4f}  "
                    f"P {entry['precision']:.4f}  R {entry['recall']:.4f}"
                )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - started, 1),
        "environment": environment_report(),
        "settings": {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou,
                     "device": args.device},
        "models": results,
    }
    (out_dir / "matrix.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dataset_names = [t[0] for t in targets]
    rows = []
    for model_name, info in results.items():
        row = {"模型": MODEL_LABELS.get(model_name, model_name)}
        for dataset_name in dataset_names:
            entry = info["datasets"].get(dataset_name, {})
            row[f"{dataset_name} mAP50"] = entry.get("map50", "")
            row[f"{dataset_name} mAP50-95"] = entry.get("map50_95", "")
        rows.append(row)

    with (out_dir / "matrix.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "summary.md").write_text(
        render_markdown(payload, dataset_names), encoding="utf-8"
    )

    print(f"\n结果已写入：{out_dir}")
    return 0


def render_markdown(payload: dict, dataset_names: list[str]) -> str:
    lines = ["# 跨域对比评测\n"]
    lines.append("> 本文件由 `scripts/compare_models.py` 自动生成，请勿手工编辑。\n")
    lines.append(f"生成时间：{payload['generated_at']}\n")
    env = payload["environment"]
    lines.append(
        f"环境：{env.get('platform')} · Python {env.get('python')} · "
        f"torch {env.get('torch')} · {env.get('gpu', 'CPU')}\n"
    )
    lines.append(
        f"评测参数：imgsz={payload['settings']['imgsz']}，"
        f"conf={payload['settings']['conf']}，iou={payload['settings']['iou']}\n"
    )

    lines.append("\n## mAP@0.5\n")
    lines.append("| 模型 | " + " | ".join(dataset_names) + " |")
    lines.append("| --- |" + " --- |" * len(dataset_names))
    for name, info in payload["models"].items():
        cells = []
        for dataset in dataset_names:
            entry = info["datasets"].get(dataset, {})
            cells.append(
                f"{entry['map50']:.4f}" if "map50" in entry else entry.get("error", "—")[:20]
            )
        lines.append(f"| {MODEL_LABELS.get(name, name)} | " + " | ".join(cells) + " |")

    lines.append("\n## mAP@0.5:0.95\n")
    lines.append("| 模型 | " + " | ".join(dataset_names) + " |")
    lines.append("| --- |" + " --- |" * len(dataset_names))
    for name, info in payload["models"].items():
        cells = []
        for dataset in dataset_names:
            entry = info["datasets"].get(dataset, {})
            cells.append(
                f"{entry['map50_95']:.4f}" if "map50_95" in entry
                else entry.get("error", "—")[:20]
            )
        lines.append(f"| {MODEL_LABELS.get(name, name)} | " + " | ".join(cells) + " |")

    # 相对基线的变化
    base = payload["models"].get("baseline")
    if base:
        lines.append("\n## 相对 Model A 基线的变化（mAP@0.5）\n")
        lines.append("| 模型 | " + " | ".join(dataset_names) + " |")
        lines.append("| --- |" + " --- |" * len(dataset_names))
        for name, info in payload["models"].items():
            if name == "baseline":
                continue
            cells = []
            for dataset in dataset_names:
                a = base["datasets"].get(dataset, {}).get("map50")
                b = info["datasets"].get(dataset, {}).get("map50")
                if a is None or b is None:
                    cells.append("—")
                else:
                    delta = b - a
                    sign = "+" if delta >= 0 else ""
                    cells.append(f"{sign}{delta:.4f}")
            lines.append(f"| {MODEL_LABELS.get(name, name)} | " + " | ".join(cells) + " |")

        lines.append(
            "\n**怎么读这张表**：只看模型在自己训练域上的提升是不够的。"
            "如果某个模型在自己的域上涨、在其它域上跌，说明它是以牺牲通用能力"
            "换取了域内表现；对本系统而言，通用人员检测能力才是主链路依赖的东西。\n"
        )

    lines.append(
        "\n> 注意：SCB-Dataset5 只标注做出特定行为的学生，不标注画面中其他人"
        "（实测覆盖率约 63%，见 `tools/datasets/scb_mapping.json`）。"
        "因此在 SCB 测试集上的指标衡量的是"能否复现该数据集的标注习惯"，"
        "**不等同于"课堂人员检测能力"**。解读时必须结合这一点。\n"
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
