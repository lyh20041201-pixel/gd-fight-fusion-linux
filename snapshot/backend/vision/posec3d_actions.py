"""SAFER-Activities PoseC3D inference on the existing COCO-17 pose stream.

The public checkpoint predicts the middle of 48 frames. Live decisions are
made only after the whole window is observed (about one second of context
delay). We resample timestamps to 25 Hz; repeated samples never count toward
the minimum eight distinct, usable pose observations. This is an adaptation
using YOLOv8n poses, not a reproduction of the paper's ViTPose-H benchmark.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn

from .posec3d_core import ResNet3dSlowOnly, PoseCompact, GeneratePoseTarget

WEIGHTS_SHA256 = '386d5580d612f9aa40801a2724945eeb4c2d1182c0f81e5e07c8ae5757470124'
CLASSES = ('stand', 'stand_activity', 'walk', 'sit', 'sit_activity',
           'sitting_down', 'getting_up', 'bend', 'unstable', 'fall',
           'lie_down', 'lying_down', 'reach', 'run', 'jump')
FALL_INDEX = CLASSES.index('fall')
BACKBONE = dict(in_channels=17, base_channels=32, num_stages=3,
                out_indices=(2,), stage_blocks=(4, 6, 3), conv1_stride=(1, 1),
                pool1_stride=(1, 1), inflate=(0, 1, 1),
                spatial_strides=(2, 2, 2), temporal_strides=(1, 1, 2))


class PoseC3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet3dSlowOnly(**BACKBONE)
        self.cls_head = nn.Module()
        self.cls_head.fc_cls = nn.Linear(512, len(CLASSES))

    def forward(self, heatmaps):
        features = self.backbone(heatmaps).mean(dim=(2, 3, 4))
        return self.cls_head.fc_cls(features)


def load_posec3d(path, device='cuda'):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError('SAFER PoseC3D checkpoint checksum mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = PoseC3D()
    model.load_state_dict(state['state_dict'], strict=True)
    return model.to(device).eval()


def pose_heatmaps(points, shape):
    """Official PoseCompact, 64x64 Resize, GeneratePoseTarget + flip test."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (48, 17, 3) or not np.isfinite(points).all():
        raise ValueError('Expected finite 48x17x3 keypoints')
    if len(shape) < 2 or min(shape[:2]) <= 0:
        raise ValueError('Invalid source shape')
    data = dict(keypoint=points[None, ..., :2].copy(),
                keypoint_score=points[None, ..., 2].copy(), img_shape=tuple(shape[:2]))
    data = PoseCompact(hw_ratio=1., allow_imgpad=True)(data)
    h, w = data['img_shape']
    data['keypoint'] *= np.array([64/w, 64/h], dtype=np.float32)
    data['img_shape'] = (64, 64)
    heatmaps = GeneratePoseTarget(double=True)(data)['imgs']
    return np.ascontiguousarray(heatmaps.reshape(2, 48, 17, 64, 64).transpose(0, 2, 1, 3, 4))


def observed_windows(clip):
    """Four overlapping past-only 1.88-second windows per live decision."""
    points = clip['keypoints'].detach().cpu().numpy()
    times = np.asarray(clip['timestamps'], dtype=np.float64)
    indices = np.asarray(clip['frame_indices'])
    if points.ndim != 4 or points.shape[2:] != (17, 3) or points.shape[1] != len(times):
        raise ValueError('Invalid pose clip shape')
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        return []
    result = []
    for end in times[-1] - np.array([1.5, 1., .5, 0.]):
        desired = end - np.arange(47, -1, -1) / 25.
        available = np.flatnonzero((times >= desired[0] - .12) & (times <= end + 1e-8))
        if len(available) < 8:
            continue
        actual = times[available]
        if actual[-1] - actual[0] < 1.7 or np.diff(actual).max() > .5:
            continue
        selected = available[np.abs(actual[:, None] - desired[None, :]).argmin(axis=0)]
        valid = (points[:, selected][:, :, 5:, 2] >= .3).sum(axis=-1) >= 4
        for track in range(len(points)):
            if len(np.unique(indices[selected][valid[track]])) < 8:
                continue
            result.append(dict(points=points[track, selected], track=track,
                               start=float(times[selected[0]]), end=float(times[selected[-1]]),
                               observed_unique=int(len(np.unique(indices[selected][valid[track]])))))
    if len({x['track'] for x in result}) > 4:
        raise ValueError('PoseC3D live trial supports at most four eligible tracks')
    return result


def score_posec3d(model, clip, shape, threshold=.3):
    windows = observed_windows(clip)
    if not windows:
        return dict(score=None, is_fall=False, usable_tracks=0, evaluated_windows=0)
    device = next(model.parameters()).device
    predictions = []
    with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == 'cuda'):
        for window in windows:
            heatmaps = torch.from_numpy(pose_heatmaps(window['points'], shape)).to(device)
            # Author test_cfg averages softmax probabilities, not logits.
            probabilities = model(heatmaps).float().softmax(-1).mean(0)
            if not torch.isfinite(probabilities).all():
                raise ValueError('Non-finite PoseC3D prediction')
            values = probabilities.cpu().numpy()
            label = int(values.argmax())
            predictions.append(dict(score=float(values[FALL_INDEX]), is_fall=label == FALL_INDEX,
                predicted_action=CLASSES[label], action_score=float(values[label]),
                probabilities=values.tolist(), track=window['track'],
                evidence_start=window['start'], evidence_end=window['end'],
                observed_unique=window['observed_unique']))
    # A qualifying fall in any track/window must not be hidden by another action.
    best = max(predictions, key=lambda p: (p['is_fall'] and p['score'] >= threshold, p['score']))
    return dict(best, usable_tracks=len({w['track'] for w in windows}), evaluated_windows=len(windows))
