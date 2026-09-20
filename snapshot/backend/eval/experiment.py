"""训练实验的配置加载、数据组装与记录。

一份配置驱动四组实验，不写四份训练代码。所有影响结果的东西
（权重、数据、划分、种子、超参、增强、环境）都会被完整记录，
保证毕业设计报告里的实验以后可以复现。

加权采样的做法：按目标比例计算各数据集的**重复倍数**，把样本路径在训练
清单里重复相应次数。这是最直白也最容易核对的加权方式 ——
清单文件摆在那里，每个数据集实际用了多少条一目了然。
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ExperimentError(RuntimeError):
    """配置或数据有问题，不能开始训练。"""


# ============================================================
# 配置
# ============================================================


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并。override 里的值覆盖 base，字典逐层合并而不是整个替换。"""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_experiment(path: Path, _seen: set[Path] | None = None) -> dict:
    """加载实验配置，处理 extends 继承。"""
    import yaml

    path = Path(path).resolve()
    seen = _seen or set()
    if path in seen:
        raise ExperimentError(f"配置继承出现循环：{path}")
    if not path.exists():
        raise ExperimentError(f"实验配置不存在：{path}")
    seen.add(path)

    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        raise ExperimentError(f"配置 {path.name} 不是合法 YAML：{exc}") from exc
    if not isinstance(config, dict):
        raise ExperimentError(f"配置 {path.name} 顶层必须是映射结构")

    parent_name = config.pop("extends", None)
    if parent_name:
        parent = load_experiment(path.parent / str(parent_name), seen)
        config = _deep_merge(parent, config)
    return config


# ============================================================
# 数据组装
# ============================================================


@dataclass
class DatasetPlan:
    """一个数据集在本次实验中的实际使用情况。"""

    name: str
    root: Path
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    weight: float = 1.0
    repeat: int = 1
    split_source: str = ""

    @property
    def effective_train(self) -> int:
        return len(self.train) * self.repeat

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "root": str(self.root),
            "split_source": self.split_source,
            "train_samples": len(self.train),
            "val_samples": len(self.val),
            "test_samples": len(self.test),
            "weight": self.weight,
            "repeat": self.repeat,
            "effective_train_samples": self.effective_train,
        }


def load_dataset_plan(
    spec: dict, project_root: Path, split_dir: Path
) -> DatasetPlan:
    """按配置读取一个数据集的划分清单，并做必要的前置检查。"""
    from tools.datasets.common import read_split

    name = str(spec.get("name") or "")
    if not name:
        raise ExperimentError("datasets 里有一项没写 name")

    root = project_root / str(spec.get("root") or "")
    if not root.exists():
        raise ExperimentError(
            f"数据集 {name} 的目录不存在：{root}\n"
            "请先按 datasets/README.md 下载并转换数据。"
        )

    prefix = str(spec.get("split_prefix") or name)
    splits = {
        part: read_split(split_dir / f"{prefix}_{part}.txt")
        for part in ("train", "val", "test")
    }
    if not splits["train"]:
        raise ExperimentError(
            f"数据集 {name} 没有训练划分清单："
            f"{split_dir / f'{prefix}_train.txt'}\n"
            "请先运行 tools/datasets/make_splits.py 生成划分。"
        )

    info_path = split_dir / f"{prefix}_split_info.json"
    split_source = ""
    if info_path.exists():
        try:
            split_source = str(
                json.loads(info_path.read_text(encoding="utf-8")).get("source", "")
            )
        except ValueError:
            split_source = ""

    # 视频数据集必须按序列划分，否则测试集里全是训练时见过的画面
    if spec.get("require_grouped_split") and split_source not in {
        "grouped_by_sequence",
        "official",
    }:
        raise ExperimentError(
            f"数据集 {name} 的划分方式是 `{split_source or '未知'}`，"
            "但它是视频数据集，必须按序列整体划分。\n"
            "随机抽帧划分会让同一段视频的帧同时出现在训练集与测试集里，"
            "指标会虚高到没有意义。\n"
            "请重新运行：tools/datasets/make_splits.py --dataset "
            f"{prefix} --root <数据集根目录>"
        )

    return DatasetPlan(
        name=name,
        root=root,
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        weight=float(spec.get("weight", 1.0)),
        split_source=split_source or "unknown",
    )


def compute_repeats(
    plans: list[DatasetPlan],
    strategy: str = "balanced",
    max_repeat: int = 4,
) -> None:
    """计算各数据集的重复倍数（就地修改 plans）。

    两个数据集样本量相差悬殊时直接拼接，样本多的那个会主导梯度，
    少的那个几乎等于没参与训练。所以要么按配置的 weight 加权，
    要么按样本量自动均衡。

    - ``balanced``：让各数据集对训练的贡献大致相当。
      以样本量最多的数据集为基准，其余按比例过采样，
      重复倍数上限为 ``max_repeat``（防止把很小的数据集重复几十遍导致过拟合）。
    - ``manual``：严格按配置里各数据集的 weight 比例过采样。
    """
    if not plans:
        return
    if len(plans) == 1:
        plans[0].repeat = 1
        return

    counts = [len(p.train) for p in plans]
    if strategy == "manual":
        weights = [max(1e-9, p.weight) for p in plans]
        # 以"每单位权重对应的样本数"最大的那个为基准
        per_weight = [c / w for c, w in zip(counts, weights)]
        base = max(per_weight)
        for plan, count, weight in zip(plans, counts, weights):
            target = base * weight
            plan.repeat = max(1, min(max_repeat, round(target / max(1, count))))
        return

    largest = max(counts)
    for plan, count in zip(plans, counts):
        if count <= 0:
            plan.repeat = 1
            continue
        plan.repeat = max(1, min(max_repeat, round(largest / count)))


def build_file_lists(
    plans: list[DatasetPlan],
    out_dir: Path,
    project_root: Path,
    max_samples_per_epoch: int = 0,
    seed: int = 42,
) -> dict[str, Path]:
    """生成 ultralytics 用的图像清单文件。

    训练清单按重复倍数过采样；验证与测试清单**绝不过采样**
    （重复样本会让指标失真）。
    """
    import random

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for part in ("train", "val", "test"):
        lines: list[str] = []
        for plan in plans:
            entries = getattr(plan, part)
            repeat = plan.repeat if part == "train" else 1
            for _ in range(repeat):
                lines.extend(
                    str((plan.root / entry).resolve()) for entry in entries
                )

        if part == "train" and max_samples_per_epoch and len(lines) > max_samples_per_epoch:
            rng = random.Random(seed)
            rng.shuffle(lines)
            lines = lines[:max_samples_per_epoch]

        path = out_dir / f"{part}.txt"
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written[part] = path
    return written


def write_data_yaml(
    out_dir: Path, lists: dict[str, Path], names: dict[int, str]
) -> Path:
    """写 ultralytics 需要的 data.yaml。"""
    import yaml

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(out_dir.resolve()),
        "train": str(lists["train"].resolve()),
        "val": str(lists["val"].resolve()) if lists["val"].stat().st_size else str(
            lists["train"].resolve()
        ),
        "names": {int(k): str(v) for k, v in names.items()},
    }
    if lists["test"].stat().st_size:
        payload["test"] = str(lists["test"].resolve())

    path = out_dir / "data.yaml"
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


# ============================================================
# 环境记录
# ============================================================


def environment_report() -> dict:
    """记录复现实验所需的全部环境信息。"""
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor() or "未知",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            report["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            )
    except ImportError:
        report["torch"] = "未安装"
    try:
        import ultralytics

        report["ultralytics"] = ultralytics.__version__
    except ImportError:
        report["ultralytics"] = "未安装"
    try:
        import numpy

        report["numpy"] = numpy.__version__
    except ImportError:
        pass
    return report


def git_revision(project_root: Path) -> dict:
    """记录当前代码版本，便于把实验结果和代码对应起来。"""
    import subprocess

    info: dict[str, Any] = {}
    try:
        info["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        info["dirty"] = bool(status)
    except Exception:  # noqa: BLE001 - 没装 git 也不该让训练失败
        info["commit"] = "unknown"
    return info
