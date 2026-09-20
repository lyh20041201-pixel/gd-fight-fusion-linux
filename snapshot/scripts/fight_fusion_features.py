"""Bounded, versioned three-stream features from original videos, never resized crops.

Downloads run in separate processes. Callers can disable networking for this entire
process. WindowSequence is lazy, including for hours-long continuous validation.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from contextlib import nullcontext, contextmanager
from functools import wraps
from pathlib import Path
import hashlib
import json
import math
import random
import shutil
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.skeleton_common import digest, sha

OUT = ROOT / 'results/fight_fusion_v1'
CACHE = ROOT / 'datasets/fight_fusion_v1/features'
POSE = ROOT / 'models/yolov8n-pose.pt'
PRETRAIN = ROOT / 'pretrained/r3d_18-b3b3357e.pth'
POSE_SHA = 'c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212'
PRETRAIN_SHA = 'b3b3357ead25631ec9c57362ff2128a92d0427e01e2cd184951a44380c3f2e9d'
PROTOCOL = dict(version=1, seconds=4., frames=32, stride_seconds=2.,
                pose_conf=.1, track_conf=.25, pose_imgsz=640, max_rois=4,
                roi_padding=.2, train_frame_windows=16, joint_conf=.01,
                minimum_free_gib=100, disk_cache_gib=128,
                rgb='112 letterbox RGB; Kinetics mean/std; fp16 frozen layer3 boundary',
                skeleton='original confidence and real-time intervals; missingness explicit',
                frame_sampling='equal annotated-event turns; exclusive window groups; deterministic within-event rotation; short events resampled with replacement')


class ResourcePause(RuntimeError):
    """Stop with a recoverable reason, never delete source data to free space."""


def replace_with_retry(source, target):
    """Windows readers can briefly deny rename; tolerate that, not other I/O errors."""
    deadline = time.monotonic() + 3.
    while True:
        try:
            Path(source).replace(target)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.05)


@contextmanager
def isolated_random_state(device):
    """A cache miss must not change the action trainer's dropout/RNG sequence."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def preserve_rng(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with isolated_random_state(self.device):
            return method(self, *args, **kwargs)
    return wrapped


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    replace_with_retry(temp, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    torch.save(value, temp)
    replace_with_retry(temp, path)


def check_space(path=ROOT, reserve=100 * 1024**3, required=0):
    free = shutil.disk_usage(Path(path).resolve()).free
    if free - required < reserve:
        raise ResourcePause(f'Free disk {free / 1024**3:.2f} GiB; required reserve {reserve / 1024**3:.0f} GiB')
    return free


def all_spans(row):
    start = float(row.get('start', 0.))
    end = float(row['end'] if 'end' in row else row['duration'])
    if not math.isfinite(start + end) or start < 0 or end <= start:
        raise ValueError(f'Invalid source interval: {row["sample_id"]}')
    if end - start <= 4:
        return [(start, end)]
    starts = np.arange(start, end - 4, 2.).tolist()
    if not starts or abs(starts[-1] - (end - 4)) > 1e-6:
        starts.append(end - 4)
    return [(float(s), float(s + 4)) for s in starts]


def window_label(row, span):
    """Only frame-annotated rows may acquire individual window labels."""
    if row.get('label_kind', 'video') != 'frame':
        return None
    if 'positive_intervals' not in row:
        raise ValueError('Frame labels require explicit, validated positive_intervals')
    return int(any(float(a) < span[1] and float(b) > span[0]
                   for a, b in row['positive_intervals']))


def event_balanced_indices(spans, positive_intervals, sample_id, epoch,
                           positive_budget, negative_budget, *, return_metadata=False):
    """Bounded, deterministic event-equal sampling for temporal annotations.

    Every positive window belongs to exactly one sampling group. A maximum
    matching first gives each separately observable event a window of its own;
    only events that cannot receive distinct windows share a group. These are
    sampling groups, not changes to the original evaluation annotations.

    Event turns, not event duration, determine sampling mass. A short event's
    window can repeat in one epoch; this is explicit loss reweighting and never
    manufactures additional observed timestamps inside that window.
    """
    if any(int(v) != v or v < 0 for v in (epoch, positive_budget, negative_budget)):
        raise ValueError('Epoch and sampling budgets must be nonnegative integers')
    spans = [(float(a), float(b)) for a, b in spans]
    intervals = [(float(a), float(b)) for a, b in positive_intervals]
    if any(not math.isfinite(a + b) or b <= a for a, b in spans + intervals):
        raise ValueError('Sampling requires finite, nonempty temporal intervals')
    if len(set(spans)) != len(spans):
        raise ValueError('Source window intervals must be unique before intentional resampling')
    if len(set(intervals)) != len(intervals):
        raise ValueError('Duplicate annotated event intervals would alter event weights')
    intervals.sort()
    pools, overlap = [], {}
    for event, (a, b) in enumerate(intervals):
        candidates = [i for i, (s, e) in enumerate(spans) if a < e and b > s]
        if not candidates:
            raise ValueError('Annotated event has no available source window')
        for i in candidates:
            s, e = spans[i]
            overlap[i, event] = min(e, b) - max(s, a)
        candidates.sort(key=lambda i: (-overlap[i, event], abs(sum(spans[i]) - a - b), i))
        pools.append(candidates)
    positive = sorted({i for values in pools for i in values})
    positive_set = set(positive)
    negative = [i for i in range(len(spans)) if i not in positive_set]
    pcount, ncount = min(int(positive_budget), len(positive)), min(int(negative_budget), len(negative))

    def permutation(values, key):
        seed = int(hashlib.sha256((str(sample_id) + ':' + key).encode()).hexdigest()[:16], 16)
        return np.random.default_rng(seed).permutation(values).tolist()

    # Iterative augmenting paths avoid recursion depth depending on event count.
    owner, assigned = {}, {}
    priority = sorted(range(len(pools)), key=lambda e: (len(pools[e]), intervals[e][1]-intervals[e][0], e))
    for root in priority:
        queue, seen_events, via_window, discovered_by, seen_windows = [root], {root}, {}, {}, set()
        free = None
        for event in queue:
            for index in pools[event]:
                if index in seen_windows:
                    continue
                seen_windows.add(index)
                discovered_by[index] = event
                if index not in owner:
                    free = index
                    break
                previous = owner[index]
                if previous not in seen_events:
                    seen_events.add(previous)
                    via_window[previous] = index
                    queue.append(previous)
            if free is not None:
                break
        if free is not None:
            while True:
                event = discovered_by[free]
                owner[free], assigned[event] = event, free
                if event == root:
                    break
                free = via_window[event]

    parent = list(range(len(pools)))
    def group(event):
        while parent[event] != event:
            parent[event] = parent[parent[event]]
            event = parent[event]
        return event
    for event in range(len(pools)):
        if event not in assigned:
            parent[group(event)] = group(owner[pools[event][0]])
    groups = {}
    for event in range(len(pools)):
        groups.setdefault(group(event), []).append(event)
    window_group = {}
    for index in positive:
        if index in owner:
            event = owner[index]
        else:
            candidates = [e for e in range(len(pools)) if (index, e) in overlap]
            event = min(candidates, key=lambda e: (-overlap[index, e],
                        abs(sum(spans[index]) - sum(intervals[e])), e))
        window_group[index] = group(event)
    group_windows = {g: permutation([i for i in positive if window_group[i] == g],
                                     'event-group:' + repr([intervals[e] for e in events]))
                     for g, events in groups.items()}
    event_order = permutation(list(range(len(pools))), 'event-order')
    group_positions = {g: [j for j, e in enumerate(event_order) if group(e) == g] for g in groups}
    draws, positive_indices = [], []
    for slot in range(pcount):
        absolute = int(epoch) * pcount + slot
        event = event_order[absolute % len(event_order)]
        g = group(event)
        visits = (absolute // len(event_order)) * len(groups[g])
        visits += sum(j < absolute % len(event_order) for j in group_positions[g])
        index = group_windows[g][visits % len(group_windows[g])]
        draws.append(dict(index=index, event=event, group_events=groups[g],
                          replacement=index in positive_indices))
        positive_indices.append(index)
    negatives = permutation(negative, 'negative-order')
    selected_negative = [negatives[(int(epoch) * ncount + j) % len(negatives)] for j in range(ncount)]
    selected = sorted(positive_indices + selected_negative)
    if not return_metadata:
        return selected
    return selected, dict(positive_draws=draws, negative_indices=selected_negative,
                          replacement_draws=pcount-len(set(positive_indices)),
                          window_event_groups={i: groups[g] for i, g in window_group.items()},
                          shared_event_groups=[events for events in groups.values() if len(events) > 1],
                          policy=PROTOCOL['frame_sampling'])


def selected_spans(row, epoch=None, full=False):
    spans = all_spans(row)
    if full or epoch is None or row.get('label_kind', 'video') != 'frame':
        return spans
    positive = sum(window_label(row, span) == 1 for span in spans)
    negative = len(spans) - positive
    pcount = min(8, positive)
    ncount = min(16 - pcount, negative)
    pcount = min(16 - ncount, positive)
    chosen = event_balanced_indices(spans, row['positive_intervals'], row['sample_id'],
                                    epoch, pcount, ncount)
    return [spans[i] for i in chosen]


def decode_window(row, span):
    """Sample actual source frames. No synthetic duration or interpolated poses."""
    cap = cv2.VideoCapture(str(row['path']))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or fps <= 0 or count <= 0:
            raise ValueError(f'Cannot decode {row["path"]}')
        first = min(count - 1, max(0, round(span[0] * fps)))
        last = min(count - 1, max(first, round(span[1] * fps) - 1))
        indices = np.linspace(first, last, 32).astype(np.int64)
        wanted = set(indices.tolist())
        if first and not cap.set(cv2.CAP_PROP_POS_FRAMES, first):
            raise ValueError('Cannot seek to source window')
        frames = {}
        for index in range(first, last + 1):
            if not cap.grab():
                raise ValueError(f'Decode interrupted at frame {index}: {row["path"]}')
            if index in wanted:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    raise ValueError(f'Cannot retrieve frame {index}')
                if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - index - 1) > 1.1:
                    raise ValueError(f'Seek timestamp mismatch at frame {index}')
                frames[index] = frame
        images = [frames[int(i)] for i in indices]
        return images, (indices / fps).tolist(), indices.tolist(), list(images[0].shape[:2])
    finally:
        cap.release()


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


class WindowSequence(Sequence):
    def __init__(self, store, row, spans):
        self.store, self.row, self.spans = store, row, spans

    def __len__(self):
        return len(self.spans)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return self.store.window(self.row, self.spans[index])


class FeatureStore:
    """Per-window disk cache and tiny CPU LRU; lazy GPU models, no test scoring."""
    def __init__(self, manifest=None, cache_root=CACHE, *, device='cuda',
                 allow_build=True, allow_partial=False, minimum_free_gib=100,
                 disk_cache_gib=128):
        self.manifest = manifest
        if isinstance(manifest, (str, Path)):
            self.manifest = json.loads(Path(manifest).read_text(encoding='utf-8'))
        if self.manifest and not allow_partial:
            if self.manifest.get('status') not in ('sealed', 'complete', 'ready'):
                raise ValueError('Formal feature preparation requires complete sealed data')
        self.device = torch.device(device)
        self.cache_root = Path(cache_root).resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.allow_build = allow_build
        self.reserve = int(minimum_free_gib * 1024**3)
        self.disk_limit = int(disk_cache_gib * 1024**3)
        self.pose = self.extractor = None
        self.verified = set()
        self.lru = OrderedDict()
        self.memory_bytes = 0
        self.memory_limit = 128 * 1024**2
        if sha(POSE) != POSE_SHA or sha(PRETRAIN) != PRETRAIN_SHA:
            raise ValueError('Base model provenance mismatch')
        self.signature = digest(dict(protocol=PROTOCOL, pose_sha=POSE_SHA, pretrained_sha=PRETRAIN_SHA,
            code={p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__), ROOT/'backend/vision/fight_fusion.py']},
            torch=torch.__version__, precision='cuda fp16 features / CPU fp32'))
        self.folder = self.cache_root / self.signature[:16]
        self.folder.mkdir(parents=True, exist_ok=True)
        self.entries = {p: (p.stat().st_size, p.stat().st_mtime) for p in self.cache_root.rglob('*.pt')}
        self.disk_bytes = sum(v[0] for v in self.entries.values())

    def get(self, row, epoch=None, full=False):
        return WindowSequence(self, row, selected_spans(row, epoch, full))

    def _evict(self, required):
        # Only individual cache files below the explicitly resolved cache root.
        # Source videos, checkpoints and arbitrary external paths are never candidates.
        if required > self.disk_limit:
            raise ResourcePause('One feature window exceeds the configured disk cache limit')
        if self.disk_bytes + required <= self.disk_limit:
            return
        for path, (size, _) in sorted(self.entries.items(), key=lambda p: p[1][1]):
            if self.cache_root not in path.resolve().parents:
                raise ValueError('Unsafe cache eviction path')
            path.unlink(missing_ok=True)
            self.entries.pop(path, None)
            self.disk_bytes -= size
            if self.disk_bytes + required <= self.disk_limit * .9:
                break

    @preserve_rng
    def _init_models(self):
        from backend.vision.fight_fusion import make_rgb_model
        if self.pose is None:
            from ultralytics import YOLO
            self.pose = YOLO(str(POSE), task='pose')
            self.pose.to(str(self.device))
        if self.extractor is None:
            self.extractor = make_rgb_model().to(self.device).eval()
            for param in self.extractor.parameters():
                param.requires_grad_(False)

    def _features(self, rgb):
        from backend.vision.fight_fusion import rgb_tensor
        context = torch.autocast('cuda') if self.device.type == 'cuda' else nullcontext()
        with torch.inference_mode(), context:
            x = rgb_tensor(rgb).unsqueeze(0).to(self.device)
            m = self.extractor
            z = m.layer3(m.layer2(m.layer1(m.stem(x))))
            boundary = z.half().float() if self.device.type == 'cuda' else z.float()
            pooled = m.avgpool(m.layer4(boundary)).flatten(1)
        # Explicit half->float boundary is reproduced by every training/scoring path.
        cached = z[0].cpu().half() if self.device.type == 'cuda' else z[0].cpu().float()
        # Clone outside inference_mode: CPU .cpu().float() can otherwise keep
        # the original inference tensor, which a trainable FC cannot save.
        return cached.clone(), pooled[0].cpu().float().clone()

    @preserve_rng
    def window(self, row, span):
        from backend.vision.fight_fusion import (extract_pose_clip, interaction_rois, crop_rgb_frames,
            rgb_array, prepare_fusion_skeleton, window_quality)
        key = digest([self.signature, row['sample_id'], row['sha256'], span])
        if key in self.lru:
            self.lru.move_to_end(key)
            data = self.lru[key]
        else:
            path = self.folder / key[:2] / (key + '.pt')
            if path.is_file():
                saved = torch.load(path, map_location='cpu', weights_only=True)
                if saved['signature'] != self.signature or saved['key'] != key:
                    raise ValueError('Feature cache signature mismatch')
                data = saved['window']
            else:
                if not self.allow_build:
                    raise FileNotFoundError(f'Features not prepared: {row["sample_id"]} {span}')
                check_space(self.cache_root, self.reserve, required=16*1024**2)
                identity = (row['path'], row['sha256'])
                if identity not in self.verified:
                    if sha(row['path']) != row['sha256']:
                        raise ValueError(f'Source changed: {row["path"]}')
                    self.verified.add(identity)
                self._init_models()
                images, times, indices, shape = decode_window(row, span)
                clip = extract_pose_clip(images, times, indices, self.pose, camera_id=str(row['sample_id']))
                clip.update(start=float(span[0]), end=float(span[1]))
                rois = interaction_rois(clip, shape, max_rois=4, padding=.2)
                full_rgb = rgb_array(images)
                gl3, gp = self._features(full_rgb)
                locals_ = [self._features(c) for c in crop_rgb_frames(images, rois)]
                rl3 = torch.stack([a for a, _ in locals_]) if locals_ else torch.empty((0, *gl3.shape), dtype=gl3.dtype)
                rp = torch.stack([b for _, b in locals_]) if locals_ else torch.empty((0, gp.shape[0]))
                data = dict(start=float(span[0]), end=float(span[1]), timestamps=times, frame_indices=indices,
                    global_layer3=gl3, global_pooled=gp, roi_layer3=rl3, roi_pooled=rp,
                    skeleton=prepare_fusion_skeleton(clip, shape), quality=window_quality(clip, shape, rois),
                    source_shape=shape, clip=clip, rgb=torch.from_numpy(full_rgb), rois=rois)
                self._evict(tensor_bytes(data) + 65536)
                atomic_torch(path, dict(signature=self.signature, key=key, window=data))
                stat = path.stat()
                self.entries[path] = (stat.st_size, stat.st_mtime)
                self.disk_bytes += stat.st_size
            size = tensor_bytes(data)
            if size < self.memory_limit:
                while self.lru and (self.memory_bytes + size > self.memory_limit or len(self.lru) >= 8):
                    _, previous = self.lru.popitem(last=False)
                    self.memory_bytes -= tensor_bytes(previous)
                self.lru[key] = data
                self.memory_bytes += size
        # Labels are kept outside immutable visual features and follow the sealed manifest.
        result = dict(data)
        result['label'] = window_label(row, span)
        return result

    def release_models(self):
        self.pose = self.extractor = None
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def close(self):
        self.release_models()
        self.lru.clear()
        self.memory_bytes = 0


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default=str(OUT/'data/manifest.json'))
    p.add_argument('--split', nargs='+', default=['train', 'branch_val'])
    p.add_argument('--limit', type=int)
    p.add_argument('--epoch', type=int, default=0)
    p.add_argument('--full', action='store_true')
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    from scripts.skeleton_common import offline
    offline()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    rows = [r for r in manifest['rows'] if r['split'] in args.split]
    if args.limit is not None:
        rows = rows[:args.limit]
    store = FeatureStore(manifest, allow_partial=args.allow_partial)
    count = 0
    try:
        for i, row in enumerate(rows):
            for _ in store.get(row, args.epoch, args.full):
                count += 1
            progress = dict(phase='prepare_features', completed=i+1, total=len(rows), windows=count,
                            signature=store.signature, disk_cache_bytes=store.disk_bytes)
            atomic_json(OUT/'feature_progress.json', progress)
            print(json.dumps(progress), flush=True)
    finally:
        store.close()


if __name__ == '__main__':
    main()
