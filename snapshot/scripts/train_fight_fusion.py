"""Offline branch training for the classroom fight fusion experiment.

Only train and branch_val supply model observations. Video labels use bounded
max-MIL; temporally annotated videos average their usable window losses so a
long source does not receive more weight merely because it has more windows.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.skeleton_round2 import choose_threshold, selection_key

BRANCHES = ("global", "roi", "skeleton_random", "skeleton_ntu")
PRETRAIN = ROOT / "pretrained/r3d_18-b3b3357e.pth"
NTU_PRETRAIN = ROOT / "models/stgcnpp/ntu60_xsub_hrnet_joint.pth"
DEFAULT_ROOT = ROOT / "results/fight_fusion_v1"
ACCUMULATION = 4
MIN_FREE_BYTES = 100 * 1024**3


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def replace_with_retry(temporary, destination):
    """Windows status readers may briefly hold the destination open."""
    deadline = time.monotonic() + 3.0
    delay = .025
    while True:
        try:
            Path(temporary).replace(destination)
            return
        except PermissionError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, .25)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temporary, path)


def check_disk(path, minimum=MIN_FREE_BYTES):
    probe = Path(path).resolve()
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < minimum:
        raise RuntimeError(f"Training paused: {free / 1024**3:.1f} GiB free; 100 GiB required")
    return free


@contextlib.contextmanager
def trainer_lock(root):
    """A script-wide lock distinct from the orchestrator's GPU lock."""
    path = Path(root) / "branch_trainer.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def rng_state():
    n = np.random.get_state()
    return dict(torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                python=random.getstate(),
                numpy=dict(name=n[0], keys=n[1].tolist(), position=n[2],
                           has_gauss=n[3], cached_gauss=n[4]))


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["name"], np.asarray(n["keys"], dtype=np.uint32),
                         n["position"], n["has_gauss"], n["cached_gauss"]))


class IsolatedWindows:
    """Lazy cache construction cannot perturb training/dropout RNG state."""
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        before = rng_state()
        try:
            return self.windows[index]
        finally:
            restore_rng(before)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def bn_state(model):
    return {name: {key: value.detach().clone() for key, value in module._buffers.items()
                   if value is not None}
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training}


def restore_bn(model, state):
    # Replace buffers, not in-place writes into values needed by autograd.
    for name, module in model.named_modules():
        for key, value in state.get(name, {}).items():
            setattr(module, key, value.clone())


def amp(device):
    return torch.autocast(device_type=device.type, enabled=device.type == "cuda",
                          cache_enabled=False)


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


def window_logit(model, window, branch, stage, device):
    if branch.startswith("skeleton"):
        item = window.get("skeleton")
        return None if item is None else model(to_device(item, device))
    key = "global" if branch == "global" else "roi"
    value = window.get(key + ("_pooled" if stage == "head" else "_layer3"))
    if value is None or value.numel() == 0:
        return None
    value = value.to(device=device, dtype=torch.float32)
    if branch == "global":
        value = value.unsqueeze(0)
    if stage != "head":
        value = model.avgpool(model.layer4(value)).flatten(1)
    logits = model.fc(value)
    # Two-class R3D models retain the project's normal/fight state-dict shape.
    scores = logits[:, 1] - logits[:, 0] if logits.shape[-1] == 2 else logits[:, 0]
    return scores.amax()


def backward_row(model, windows, row, forward, criterion, scaler=None, divisor=ACCUMULATION):
    """Exact max-MIL or frame-label mean with one window's activation graph.

    A no-grad scan records dropout and mutable BN state. Each selected window is
    replayed and immediately backpropagated, including tied MIL maxima. Unknown
    windows never become negative examples. BN/RNG finish at the scan state.
    """
    observations = []
    frame_labels = row.get("label_kind", "video") == "frame"
    maximum = None
    with torch.no_grad():
        for index, window in enumerate(windows):
            label = window.get("label") if frame_labels else row["label"]
            if label not in (0, 1):
                continue
            before = (rng_state(), bn_state(model))
            value = forward(window)
            if value is None:
                continue
            score = float(value.detach().float())
            if not np.isfinite(score):
                raise RuntimeError("Nonfinite branch training logit")
            record = (index, score, int(label), before)
            if frame_labels:
                observations.append(record)
            elif maximum is None or score > maximum:
                maximum = score
                observations = [record]
            elif score == maximum:
                observations.append(record)
    if not observations:
        return None
    after_rng, after_bn = rng_state(), bn_state(model)
    count = len(observations)
    total = 0.0
    try:
        for index, _, label, (before_rng, before_bn) in observations:
            restore_rng(before_rng)
            restore_bn(model, before_bn)
            value = forward(windows[index])
            if value is None:
                raise RuntimeError("Branch eligibility changed during MIL replay")
            value = value.float()
            loss = criterion(value, value.new_tensor(float(label)))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite branch training loss")
            total += float(loss.detach()) / count
            gradient_loss = loss / (count * divisor)
            if scaler is None:
                gradient_loss.backward()
            else:
                scaler.scale(gradient_loss).backward()
    finally:
        restore_rng(after_rng)
        restore_bn(model, after_bn)
    return total


def stages_for(branch, smoke=False, max_epochs=None):
    if smoke:
        budget = max_epochs or 1
        if branch == "skeleton_random":
            return [("finetune", budget)]
        return [("head", 1), ("finetune", budget - 1)] if budget > 1 else [("head", 1)]
    return [("finetune", 50)] if branch == "skeleton_random" else [("head", 10), ("finetune", 40)]


def configure_optimizer(model, branch, stage):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if branch in ("global", "roi"):
        for parameter in model.fc.parameters():
            parameter.requires_grad_(True)
        if stage == "finetune":
            for parameter in model.layer4.parameters():
                parameter.requires_grad_(True)
        groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                   "lr": 1e-3 if stage == "head" else 1e-4}]
    elif branch == "skeleton_random":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        groups = [{"params": list(model.parameters()), "lr": 1e-3}]
    else:
        head = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
        for parameter in head:
            parameter.requires_grad_(True)
        groups = [{"params": head, "lr": 1e-3 if stage == "head" else 1e-4}]
        if stage == "finetune":
            for parameter in model.backbone.parameters():
                parameter.requires_grad_(True)
            groups.append({"params": list(model.backbone.parameters()), "lr": 1e-5})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def training_mode(model, branch, stage):
    model.train()
    if branch in ("global", "roi"):
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    elif branch == "skeleton_ntu" and stage == "head":
        model.backbone.eval()


def training_rows(manifest):
    rows = manifest["rows"]
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "branch_val"]
    for name, subset in (("train", train), ("branch_val", validation)):
        if {row["label"] for row in subset} != {0, 1}:
            raise ValueError(f"{name} requires both source classes")
        if len({row["sample_id"] for row in subset}) != len(subset):
            raise ValueError(f"Duplicate sample ID in {name}")
    for key in ("sample_id", "sha256", "group"):
        if {row[key] for row in train} & {row[key] for row in validation}:
            raise ValueError(f"train/branch_val {key} leakage")
    return train, validation


def evaluate(model, rows, store, branch, stage, device, progress=None):
    model.eval()
    scores = []
    details = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            values = []
            window_details = []
            for window in IsolatedWindows(store.get(row, full=True)):
                with amp(device):
                    value = window_logit(model, window, branch, stage, device)
                score = None if value is None else float(value.float().sigmoid())
                if score is not None and not np.isfinite(score):
                    raise RuntimeError("Nonfinite validation score")
                if score is not None:
                    values.append(score)
                window_details.append(dict(start=window["start"], end=window["end"],
                                           label=window.get("label"), score=score))
            score = max(values) if values else None
            scores.append(score)
            details.append(dict(sample_id=row["sample_id"], dataset=row["dataset"],
                                label=row["label"], score=score, windows=window_details))
            if progress and ((index + 1) % 25 == 0 or index + 1 == len(rows)):
                progress(index + 1, len(rows))
    return scores, details


def checkpoint(path, model, optimizer, scaler, state, config_signature):
    check_disk(path)
    atomic_torch(path, dict(model=model.state_dict(),
                           optimizer=optimizer.state_dict() if optimizer else None,
                           scaler=scaler.state_dict() if scaler else None,
                           runner_state=copy.deepcopy(state), rng=rng_state(),
                           config_signature=config_signature))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def smoke_subset(rows, per_class=2):
    return [row for label in (0, 1)
            for row in [r for r in rows if r["label"] == label][:per_class]]


def train_locked(args, manifest, manifest_path, destination):
    from scripts.fight_fusion_features import FeatureStore
    validate_manifest_seal(manifest, manifest_path, smoke=args.smoke)
    kwargs = {"cache_root": Path(args.cache_root)} if args.cache_root else {}
    kwargs.update(device=args.device, allow_partial=args.smoke)
    store = FeatureStore(manifest, **kwargs)
    try:
        _train_with_store(args, manifest, manifest_path, destination, store)
    finally:
        store.close()


def validate_manifest_seal(manifest, path, smoke=False):
    if smoke:
        return
    if manifest.get("status") != "sealed" or manifest.get("training_allowed") is not True:
        raise ValueError("Production training requires a sealed, training-authorized manifest")
    seal_path = Path(path).with_suffix(".seal.json")
    if not seal_path.is_file():
        raise ValueError("Manifest source seal is missing")
    seal = read_json(seal_path)
    if seal.get("manifest_sha256") != file_sha(path):
        raise ValueError("Manifest source seal hash mismatch")


def _train_with_store(args, manifest, manifest_path, destination, store):
    from backend.vision.fight_fusion import make_rgb_model, FusionSkeletonModel

    train, validation = training_rows(manifest)
    if args.smoke:
        train, validation = smoke_subset(train), smoke_subset(validation)
    device = torch.device(args.device)
    stage_limits = stages_for(args.branch, args.smoke, args.max_epochs)
    code_paths = ["scripts/train_fight_fusion.py", "scripts/fight_fusion_features.py",
                  "backend/vision/fight_fusion.py", "backend/vision/stgcnpp_actions.py",
                  "backend/vision/stgcnpp_backbone.py", "scripts/skeleton_round2.py"]
    config = dict(version=1, branch=args.branch, seed=args.seed, smoke=args.smoke,
                  manifest_sha256=file_sha(manifest_path), stages=stage_limits,
                  manifest_seal_sha256=(None if args.smoke else file_sha(Path(manifest_path).with_suffix(".seal.json"))),
                  train_sample_ids=[row["sample_id"] for row in train],
                  validation_sample_ids=[row["sample_id"] for row in validation],
                  initialization=("local Kinetics-400" if args.branch in ("global", "roi") else
                                  "random ST-GCN++" if args.branch == "skeleton_random" else "local NTU60 HRNet Joint"),
                  pretrained_sha256=(None if args.branch == "skeleton_random" else
                                     file_sha(PRETRAIN if args.branch in ("global", "roi") else NTU_PRETRAIN)),
                  feature_signature=getattr(store, "signature", None),
                  code_hashes={p: file_sha(ROOT / p) for p in code_paths},
                  device=device.type, torch=str(torch.__version__),
                  accumulation=ACCUMULATION, optimizer="AdamW", weight_decay=1e-4,
                  gradient_clip=5., patience=8, validation_normal_fpr_limit=.05,
                  numerical="FP16 autocast / FP32 loss and sigmoid / deterministic / TF32 disabled",
                  loss="video max-MIL; frame-labelled windows mean per source; unknown omitted from loss",
                  selection="branch_val source max score, recall at FPR<=5%, lower FPR, macro F1; unknown retained",
                  data_access=["train", "branch_val"], test_accessed=False)
    config_signature = signature(config)
    config_path = destination / "config.json"
    if config_path.exists() and signature(read_json(config_path)) != config_signature:
        raise ValueError("Existing training config differs; use a new experiment directory")
    atomic_json(config_path, config)
    selection_path = destination / "selection.json"
    if selection_path.exists():
        old = read_json(selection_path)
        if old["config_signature"] != config_signature or old["sha256"] != file_sha(destination / "selected_best.pt"):
            raise ValueError("Completed branch provenance mismatch")
        print("ALREADY_COMPLETE", args.branch, args.seed, flush=True)
        return
    seed_everything(args.seed)
    model = make_rgb_model() if args.branch in ("global", "roi") else FusionSkeletonModel()
    if args.branch == "skeleton_ntu":
        model.load_ntu60()
    model.to(device)
    positive = sum(row["label"] for row in train)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train) - positive) / positive, device=device))
    state = dict(stage_index=0, stage_epoch=0, epochs_total=0, cursor=0, order=[],
                 patience=0, loss_sum=0., used=0, unknown=0, best_key=None,
                 stage_best_key=None, optimizer_steps=0, history=[], phase="train")
    resume_path = destination / "resume.pt"
    saved = None
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if saved["config_signature"] != config_signature:
            raise ValueError("Resume signature mismatch")
        state = saved["runner_state"]
        model.load_state_dict(saved["model"], strict=True)
    while state["stage_index"] < len(stage_limits):
        stage, limit = stage_limits[state["stage_index"]]
        optimizer = configure_optimizer(model, args.branch, stage)
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        if saved:
            if saved["optimizer"] is not None:
                optimizer.load_state_dict(saved["optimizer"])
                scaler.load_state_dict(saved["scaler"])
            restore_rng(saved["rng"])
            saved = None
        last_save = time.monotonic()
        while state["stage_epoch"] < limit and state["patience"] < 8:
            started = time.monotonic()
            training_mode(model, args.branch, stage)
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            if not state["order"]:
                generator = torch.Generator().manual_seed(args.seed * 10000 + state["epochs_total"])
                state["order"] = torch.randperm(len(train), generator=generator).tolist()
            order = state["order"]
            for position in range(state["cursor"], len(order)):
                row = train[order[position]]
                windows = IsolatedWindows(store.get(row, epoch=state["epochs_total"], full=False))

                def forward(window):
                    with amp(device):
                        return window_logit(model, window, args.branch, stage, device)

                loss = backward_row(model, windows, row, forward, criterion, scaler)
                state["cursor"] = position + 1
                if loss is None:
                    state["unknown"] += 1
                else:
                    pending += 1
                    state["used"] += 1
                    state["loss_sum"] += loss
                if pending == ACCUMULATION or (position == len(order) - 1 and pending):
                    scaler.unscale_(optimizer)
                    if pending != ACCUMULATION:
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.mul_(ACCUMULATION / pending)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
                    state["optimizer_steps"] += 1
                if (position + 1) % 25 == 0 or position + 1 == len(order):
                    report = dict(phase="train", branch=args.branch, seed=args.seed, stage=stage,
                                  epoch=state["epochs_total"] + 1, completed=position + 1,
                                  total=len(order), used=state["used"], unknown=state["unknown"])
                    atomic_json(destination / "progress.json", report)
                    print(json.dumps(report), flush=True)
                if pending == 0 and time.monotonic() - last_save >= 60:
                    checkpoint(resume_path, model, optimizer, scaler, state, config_signature)
                    last_save = time.monotonic()
            if not state["used"]:
                raise RuntimeError("No usable training sources")
            state["phase"] = "validation"
            checkpoint(resume_path, model, optimizer, scaler, state, config_signature)

            def validation_progress(completed, total):
                atomic_json(destination / "progress.json", dict(phase="branch_validation",
                            branch=args.branch, seed=args.seed, epoch=state["epochs_total"] + 1,
                            stage=stage, completed=completed, total=total))

            scores, details = evaluate(model, validation, store, args.branch, stage, device, validation_progress)
            threshold, metrics = choose_threshold([row["label"] for row in validation], scores, limit=.05)
            key = selection_key(metrics)
            record = dict(stage=stage, stage_epoch=state["stage_epoch"] + 1,
                          epoch=state["epochs_total"] + 1, threshold=threshold, validation=metrics,
                          loss=state["loss_sum"] / state["used"], used=state["used"],
                          unknown=state["unknown"], seconds=time.monotonic() - started)
            state["history"].append(record)
            payload = dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                           config=config, config_signature=config_signature, **record)
            if state["stage_best_key"] is None or key > tuple(state["stage_best_key"]):
                state["stage_best_key"] = list(key)
                state["patience"] = 0
                atomic_torch(destination / f"best_{stage}.pt", payload)
            else:
                state["patience"] += 1
            if state["best_key"] is None or key > tuple(state["best_key"]):
                state["best_key"] = list(key)
                atomic_torch(destination / "best.pt", payload)
                atomic_json(destination / "best_validation_predictions.json", dict(threshold=threshold, rows=details))
            state.update(stage_epoch=state["stage_epoch"] + 1, epochs_total=state["epochs_total"] + 1,
                         cursor=0, order=[], loss_sum=0., used=0, unknown=0, phase="train")
            checkpoint(resume_path, model, optimizer, scaler, state, config_signature)
            atomic_json(destination / "run_record.json", dict(status="training", config_signature=config_signature, **state))
            print("EPOCH_COMPLETE", args.branch, args.seed, record["epoch"], metrics, flush=True)
        best_stage = torch.load(destination / f"best_{stage}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(best_stage["state_dict"], strict=True)
        state.update(stage_index=state["stage_index"] + 1, stage_epoch=0, patience=0,
                     cursor=0, order=[], stage_best_key=None, phase="train")
        checkpoint(resume_path, model, None, None, state, config_signature)
    best = torch.load(destination / "best.pt", map_location="cpu", weights_only=True)
    atomic_torch(destination / "selected_best.pt", best)
    atomic_json(selection_path, dict(status="complete", branch=args.branch, seed=args.seed,
                config_signature=config_signature, epochs_trained=state["epochs_total"],
                selected_epoch=best["epoch"], selected_stage=best["stage"], threshold=best["threshold"],
                validation=best["validation"], sha256=file_sha(destination / "selected_best.pt"),
                checkpoint=str(destination / "selected_best.pt"), test_accessed=False, smoke=args.smoke))
    atomic_json(destination / "progress.json", dict(phase="complete", branch=args.branch, seed=args.seed,
                                                  epochs_trained=state["epochs_total"], smoke=args.smoke))
    atomic_json(destination / "run_record.json", dict(status="complete", config_signature=config_signature, **state))


def run(args):
    if args.max_epochs is not None and (not args.smoke or args.max_epochs < 1):
        raise ValueError("--max-epochs is positive and smoke-only; production budgets are fixed")
    if args.device != "cuda" and not args.smoke:
        raise ValueError("Production branch training requires CUDA")
    from scripts.skeleton_common import offline
    offline()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Local CUDA required; no automatic installation")
    root = Path(args.output_root).resolve()
    check_disk(root)
    destination = root / ("smoke" if args.smoke else "branches") / args.branch / f"seed{args.seed}"
    manifest_path = Path(args.manifest).resolve()
    manifest = read_json(manifest_path)
    with trainer_lock(root):
        destination.mkdir(parents=True, exist_ok=True)
        train_locked(args, manifest, manifest_path, destination)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--branch", choices=BRANCHES, required=True)
    result.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    result.add_argument("--manifest", default=str(DEFAULT_ROOT / "data/manifest.json"))
    result.add_argument("--output-root", default=str(DEFAULT_ROOT))
    result.add_argument("--cache-root")
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--max-epochs", type=int)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
