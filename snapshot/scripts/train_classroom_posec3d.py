"""Reproducible frozen-backbone PoseC3D head adaptation on Le2i lecture room.

An exploratory single-actor, single-room experiment. Never changes live config.
Test actors are excluded from selection. Unknown labels/poses remain in event
denominators. Two fall events in the held-out room test cannot establish safety.
"""
from pathlib import Path
import sys
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
from collections import Counter
import copy
import hashlib
import json
import random
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from backend.vision.live_actions import PoseWindows
from backend.vision.posec3d_actions import load_posec3d, pose_heatmaps, CLASSES, FALL_INDEX
from scripts.prepare_classroom_posec3d import DATA, OUT, sha, write

BASE = ROOT / 'models/safer_posec3d/non-wheelchair-epoch_44.pth'
CANDIDATE = ROOT / 'models/classroom_posec3d/le2i_head_pilot.pth'
POLICY = dict(version=1, seed=42, method='Frozen PoseC3D backbone; fit existing 15-class linear head',
    objective='Video/binary-class balanced BCE on fall vs summed non-fall; conditional non-fall distillation (0.1)',
    optimizer='AdamW lr=0.0001 weight_decay=0.0001; max 150 full-batch epochs; patience 25',
    selection='Lowest validation balanced binary NLL; includes unchanged epoch zero; test unused',
    labels='OmniFall midpoint label: 1 dynamic falling positive; 0,2..8 negative; 9 other unknown/excluded from loss',
    input='Native source fps, YOLOv8n-pose COCO-17; past 48/25s window; 0.25s stride; original+flip heatmaps',
    scope='Single actor: most usable observed poses within current window; minimum 8 unique observations',
    decision='Unchanged fall argmax and probability >=0.3',
    event_matching='One alarm center in annotated dynamic fall interval; causal alarm time is center+0.94s; pre-fall/late alarms reported separately',
    deployment='Offline only. No production promotion or cross-scene claim from this single-room test')


def setup():
    sys.stdout.reconfigure(encoding='utf-8')
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(3); cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def extract_pose(row, detector):
    path = DATA / 'pose' / f"video_{row['id']:02}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        assert str(z['video_sha256']) == row['video_sha256']
        return path
    capture = cv2.VideoCapture(str(ROOT / row['video']))
    observations = []
    engine = PoseWindows(detector)
    index = 0
    while True:
        batch = []
        for _ in range(8):
            ok, frame = capture.read()
            if not ok:
                break
            batch.append((index / row['fps'], frame)); index += 1
        if not batch:
            break
        clip = engine.clip(str(row['id']), batch, batch)
        joints = clip['keypoints'].numpy(); boxes = clip['boxes'].numpy()
        for j in range(len(batch)):
            observations.append([(tid, joints[k, j], boxes[k, j]) for k, tid in enumerate(clip['track_ids'])
                if np.any(joints[k, j, :, 2] > 0)])
    capture.release()
    assert index == row['frames']
    ids = sorted({tid for obs in observations for tid, _, _ in obs})
    lookup = {tid: j for j, tid in enumerate(ids)}
    joints = np.zeros((len(ids), index, 17, 3), np.float32)
    boxes = np.zeros((len(ids), index, 4), np.float32)
    for j, obs in enumerate(observations):
        for tid, points, box in obs:
            joints[lookup[tid], j] = points; boxes[lookup[tid], j] = box
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, keypoints=joints, boxes=boxes, track_ids=np.array(ids),
        times=np.arange(index)/row['fps'], frame_indices=np.arange(index), shape=row['shape'],
        video_sha256=row['video_sha256'], pose_sha256=sha(ROOT / 'models/yolov8n-pose.pt'))
    print(f"pose {row['id']:02} {index} frames, {len(ids)} track fragments", flush=True)
    return path


def midpoint_label(row, center):
    if row.get('reviewed_normal'):
        return 0
    labels = {int(s['label']) for s in row['spans'] if s['start'] <= center < s['end']}
    if len(labels) != 1 or 9 in labels:
        return -1
    return int(next(iter(labels)) == 1)


def feature_windows(row, model):
    path = DATA / 'features' / f"video_{row['id']:02}.npz"
    meta_path = path.with_suffix('.json')
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        assert meta['source_sha256'] == row['video_sha256'] and meta['base_sha256'] == sha(BASE)
        return
    z = np.load(DATA / 'pose' / f"video_{row['id']:02}.npz", allow_pickle=False)
    times = z['times']; points = z['keypoints']; frame_indices = z['frame_indices']
    ends = sorted(set([*np.arange(1.88, float(times[-1]), .25), float(times[-1])]))
    features = []; windows = []
    for end in ends:
        desired = end - np.arange(47, -1, -1)/25.
        indices = np.flatnonzero((times >= desired[0]-.04) & (times <= end+1e-8))
        w = dict(end=float(end), center=float(end-.94), label=midpoint_label(row, end-.94),
                 feature_index=-1, reason='pose_unknown')
        if len(indices) >= 8 and times[indices[-1]]-times[indices[0]] >= 1.7:
            selected = indices[np.abs(times[indices, None]-desired[None]).argmin(0)]
            sampled = points[:, selected]
            valid = (sampled[:, :, 5:, 2] >= .3).sum(-1) >= 4
            counts = [len(np.unique(frame_indices[selected][v])) for v in valid]
            if counts and max(counts) >= 8:
                eligible = [k for k, n in enumerate(counts) if n >= 8]
                chosen = eligible if row.get('multi_person') else [max(eligible, key=lambda k: (counts[k], float(points[k, selected, :, 2].mean())))]
                for primary in chosen:
                    with torch.inference_mode(), torch.autocast('cuda'):
                        heat = torch.from_numpy(pose_heatmaps(points[primary, selected], z['shape'])).cuda()
                        feature = model.backbone(heat).mean((2, 3, 4)).float().cpu().numpy()
                    assert feature.shape == (2, 512) and np.isfinite(feature).all()
                    item = dict(w, feature_index=len(features), reason=None, observed_unique=counts[primary],
                                track_id=int(z['track_ids'][primary]))
                    features.append(feature)
                    windows.append(item)
                continue
        windows.append(w)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=np.stack(features) if features else np.zeros((0, 2, 512), np.float32))
    write(meta_path, dict(windows=windows, source_sha256=row['video_sha256'], base_sha256=sha(BASE)))
    print(f"features {row['id']:02}: {len(features)}/{len(windows)} usable", flush=True)


def load_features(rows, split):
    feats = []; labels = []; videos = []
    for row in rows:
        if row['split'] != split:
            continue
        base = DATA / 'features' / f"video_{row['id']:02}"
        z = np.load(base.with_suffix('.npz'), allow_pickle=False)['features']
        windows = json.loads(base.with_suffix('.json').read_text(encoding='utf-8'))['windows']
        for w in windows:
            if w['feature_index'] >= 0 and w['label'] >= 0:
                feats.append(z[w['feature_index']]); labels.append(w['label']); videos.append(row.get('source_group',str(row['id'])))
    counts = Counter(zip(videos, labels))
    n_videos = {label: len({vid for vid, lab in counts if lab == label}) for label in (0, 1)}
    assert all(n_videos.values()), n_videos
    weights = [1/(2*n_videos[lab]*counts[vid, lab]) for vid, lab in zip(videos, labels)]
    return torch.tensor(np.stack(feats)), torch.tensor(labels), torch.tensor(weights, dtype=torch.float32)


def binary_nll(logits, target, weights):
    # Match evaluation's average of flip probabilities (not average logits).
    probs = logits.float().softmax(-1).mean(1)[:, FALL_INDEX].clamp(1e-7, 1-1e-7)
    return (F.binary_cross_entropy(probs, target.float(), reduction='none') * weights).sum() / weights.sum()


def evaluate(rows, head, split):
    predictions = []; metrics = Counter(); latencies = []
    for row in rows:
        if row['split'] != split:
            continue
        base = DATA / 'features' / f"video_{row['id']:02}"
        features = torch.from_numpy(np.load(base.with_suffix('.npz'))['features'])
        windows = json.loads(base.with_suffix('.json').read_text(encoding='utf-8'))['windows']
        with torch.no_grad():
            probabilities = head(features).softmax(-1).mean(1).numpy()
        events = [s for s in row['spans'] if s['label'] == 1]
        alarms = []
        for w in windows:
            if w['feature_index'] < 0:
                metrics['unknown_windows'] += 1
                continue
            p = probabilities[w['feature_index']]
            is_fall = int(p.argmax()) == FALL_INDEX and p[FALL_INDEX] >= .3
            if w['label'] >= 0:
                metrics['window_'+('tp' if is_fall and w['label'] else 'fp' if is_fall else 'fn' if w['label'] else 'tn')] += 1
            if is_fall:
                alarms.append(dict(center=w['center'], decision_time=w['end'], score=float(p[FALL_INDEX])))
        matches = []
        for event in events:
            matching = [a for a in alarms if event['start'] <= a['center'] < event['end']]
            matches.append(bool(matching))
            if matching:
                latencies.append(matching[0]['decision_time']-event['start'])
        metrics['fall_events'] += len(events); metrics['detected_events'] += sum(matches)
        if not events:
            metrics['normal_videos'] += 1; metrics['normal_videos_with_alarm'] += bool(alarms)
            metrics['normal_seconds'] += row['duration']
        else:
            metrics['fall_videos_with_unmatched_alarm'] += any(not any(s['start'] <= a['center'] < s['end'] for s in events) for a in alarms)
        predictions.append(dict(video_id=row['id'], events=events, matched=matches, alarms=alarms,
            usable_windows=sum(w['feature_index'] >= 0 for w in windows), total_windows=len(windows)))
    stats = dict(metrics)
    stats['recall'] = metrics['detected_events']/metrics['fall_events'] if metrics['fall_events'] else None
    stats['normal_video_false_positive_rate'] = metrics['normal_videos_with_alarm']/metrics['normal_videos'] if metrics['normal_videos'] else None
    stats['median_detection_delay_s'] = float(np.median(latencies)) if latencies else None
    return dict(summary=stats, predictions=predictions)


def train(rows, model):
    x, y, weights = load_features(rows, 'train')
    vx, vy, vw = load_features(rows, 'validation')
    head = copy.deepcopy(model.cls_head.fc_cls).cpu().float()
    original_head = copy.deepcopy(head).eval()
    for p in original_head.parameters():
        p.requires_grad_(False)
    nonfall = [i for i in range(15) if i != FALL_INDEX]
    with torch.no_grad():
        teacher = original_head(x)[:, :, nonfall].softmax(-1)
        best_loss = float(binary_nll(head(vx), vy, vw))
    best = copy.deepcopy(head.state_dict()); best_epoch = 0; history = []
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-4)
    for epoch in range(1, 151):
        optimizer.zero_grad(set_to_none=True)
        logits = head(x)
        bce = binary_nll(logits, y, weights)
        kl = F.kl_div(logits[:, :, nonfall].log_softmax(-1), teacher, reduction='none').sum(-1).mean(-1)
        loss = bce + .1*(kl*weights).sum()/weights.sum()
        assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 5); optimizer.step()
        with torch.no_grad():
            validation_loss = float(binary_nll(head(vx), vy, vw))
        history.append(dict(epoch=epoch, loss=float(loss.detach()), validation_binary_nll=validation_loss))
        if validation_loss < best_loss-1e-7:
            best_loss = validation_loss; best_epoch = epoch; best = copy.deepcopy(head.state_dict())
        if epoch-best_epoch >= 25:
            break
    head.load_state_dict(best)
    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    state = {k:v.cpu() for k,v in model.state_dict().items()}
    state.update({'cls_head.fc_cls.'+k:v for k,v in best.items()})
    torch.save(dict(state_dict=state, classes=list(CLASSES), base_sha256=sha(BASE), policy=POLICY,
                    selected_epoch=best_epoch, threshold=.3, offline_pilot=True), CANDIDATE)
    result = dict(policy=POLICY, base_sha256=sha(BASE), manifest_sha256=sha(OUT/'manifest.json'),
        checkpoint=str(CANDIDATE), checkpoint_sha256=sha(CANDIDATE), selected_epoch=best_epoch,
        train_windows=len(x), validation_windows=len(vx), selected_validation_nll=best_loss,
        history=history, splits={s:[r['id'] for r in rows if r['split']==s] for s in ('train','validation','test')},
        before={s:evaluate(rows,original_head,s) for s in ('validation','test')},
        after={s:evaluate(rows,head,s) for s in ('validation','test')})
    write(OUT/'experiment.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('history','before','after')},ensure_ascii=False),flush=True)
    for stage in ('before','after'):
        for split in ('validation','test'):
            print(stage,split,result[stage][split]['summary'],flush=True)


def main():
    setup()
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['pose','features','train','all'],default='all')
    args=parser.parse_args()
    rows=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))['videos']
    write(OUT/'policy.json',POLICY)
    if args.stage in ('pose','all'):
        from ultralytics import YOLO
        detector=YOLO(str(ROOT/'models/yolov8n-pose.pt'),task='pose')
        for row in rows:
            extract_pose(row,detector)
        del detector
        torch.cuda.empty_cache()
    if args.stage in ('features','train','all'):
        model=load_posec3d(BASE)
        if args.stage in ('features','all'):
            for row in rows:
                feature_windows(row,model)
        if args.stage in ('train','all'):
            train(rows,model)


if __name__=='__main__':
    main()
