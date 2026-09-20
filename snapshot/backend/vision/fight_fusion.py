"""Shared, local-only inputs and inference for the three-branch fight experiment.

All branches describe the same source window. Missing pose is an observation
quality signal, never a reason to skip full-frame RGB. This module does not
change the production action manifest or import the live service.
"""
from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .stgcnpp_actions import STGCNPPActionModel

KINETICS_PATH = Path(__file__).resolve().parents[2] / 'pretrained/r3d_18-b3b3357e.pth'
BRANCH_NAMES = ('global', 'roi', 'skeleton')
QUALITY_NAMES = (
    'unique_frame_fraction', 'box_pair_coverage', 'pose_pair_coverage',
    'mean_joint_confidence', 'person_height_ratio', 'track_continuity',
    'roi_area_fraction', 'roi_count_fraction',
)
FUSION_FEATURE_NAMES = (
    tuple(f'{name}_logit' for name in BRANCH_NAMES)
    + tuple(f'{name}_available' for name in BRANCH_NAMES)
    + QUALITY_NAMES
    + tuple(f'{name}_logit_x_{quality}' for name in BRANCH_NAMES for quality in QUALITY_NAMES)
)


def _shape(source_shape):
    h, w = int(source_shape[0]), int(source_shape[1])
    if h <= 0 or w <= 0:
        raise ValueError('Positive source image dimensions required')
    return h, w


def _clip_arrays(clip):
    k = torch.as_tensor(clip['keypoints'], dtype=torch.float32).detach().cpu()
    boxes = torch.as_tensor(clip['boxes'], dtype=torch.float32).detach().cpu()
    times = torch.as_tensor(clip['timestamps'], dtype=torch.float64).detach().cpu()
    indices = torch.as_tensor(clip['frame_indices'], dtype=torch.int64).detach().cpu()
    if k.ndim != 4 or k.shape[2:] != (17, 3):
        raise ValueError('keypoints must have shape [people,time,17,3]')
    m, t = k.shape[:2]
    if boxes.shape != (m, t, 4) or times.shape != (t,) or indices.shape != (t,):
        raise ValueError('Clip arrays must have matching people and time dimensions')
    if not torch.isfinite(k).all() or not torch.isfinite(boxes).all() or not torch.isfinite(times).all():
        raise ValueError('Nonfinite input observations')
    if t > 1 and ((times[1:] < times[:-1]).any() or (indices[1:] < indices[:-1]).any()):
        raise ValueError('Window observations must be chronological')
    if t > 1 and ((times[1:] == times[:-1]) != (indices[1:] == indices[:-1])).any():
        raise ValueError('Source frame identities and timestamps must describe the same repetitions')
    if (k[..., 2] < 0).any() or (k[..., 2] > 1).any():
        raise ValueError('Keypoint confidence must be within [0,1]')
    # Relative time avoids float32 precision loss for Unix timestamps.
    times = (times - times[0]).float() if t else times.float()
    return k, boxes, times, indices


def _unique_mask(indices):
    first = torch.ones(len(indices), dtype=torch.bool, device=indices.device)
    if len(indices) > 1:
        first[1:] = indices[1:] != indices[:-1]
    return first


def prepare_fusion_skeleton(clip, source_shape):
    """COCO17 xy/confidence; >=4 real frames, >=2 body joints at confidence .1.

    Only xy coordinates with confidence <.01 are zeroed. Confidence itself is
    preserved, including very weak detections, for the learned temporal model.
    """
    h, w = _shape(source_shape)
    k, boxes, times, indices = _clip_arrays(clip)
    valid_boxes = (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])
    valid = ((k[:, :, 5:, 2] >= .1).sum(-1) >= 2) & valid_boxes
    usable = torch.tensor([len(torch.unique(indices[v])) >= 4 for v in valid], dtype=torch.bool)
    k = k[usable].clone()
    boxes, valid = boxes[usable], valid[usable]
    centers = (boxes[..., :2] + boxes[..., 2:]) / 2
    scales = [torch.linalg.vector_norm(b[v, 2:] - b[v, :2], dim=-1).median().clamp_min(1)
              for b, v in zip(boxes, valid)]
    scale = torch.stack(scales) if scales else torch.empty(0)
    xy = (k[..., :2] - torch.tensor([w / 2, h / 2])) / torch.tensor([w / 2, h / 2])
    xy = torch.where((k[..., 2] >= .01)[..., None], xy, 0)
    features = torch.cat([xy, k[..., 2:3]], -1).permute(0, 3, 1, 2).contiguous()
    return dict(features=features, valid=valid, centers=centers, scale=scale,
                times=times, frame_indices=indices)


class FusionSkeletonModel(STGCNPPActionModel):
    """Symmetric per-pair ST-GCN++ with four distinct shared observations."""
    def __init__(self, task='fight'):
        if task != 'fight':
            raise ValueError('FusionSkeletonModel is a fight model')
        super().__init__(task)

    def forward(self, item):
        features, valid = item['features'], item['valid']
        if len(features) < 2:
            return None
        pairs = torch.triu_indices(len(features), len(features), 1, device=features.device)
        shared = valid[pairs[0]] & valid[pairs[1]]
        first = _unique_mask(item['frame_indices'])
        good = (shared & first[None]).sum(-1) >= 4
        pairs, shared = pairs[:, good], shared[good]
        if not pairs.shape[1]:
            return None
        z = self.backbone(features.permute(0, 2, 3, 1).unsqueeze(0))[0].mean(-1)
        target_t = z.shape[-1]
        outputs = []
        dt = item['times'][1:] - item['times'][:-1]
        for pair, mask in zip(pairs.split(128, dim=1), shared.split(128)):
            i, j = pair
            scale = ((item['scale'][i] + item['scale'][j]) / 2).clamp_min(1)
            delta = (item['centers'][i] - item['centers'][j]) / scale[:, None, None]
            velocity = torch.zeros_like(delta)
            velocity[:, 1:] = (delta[:, 1:] - delta[:, :-1]) / dt.clamp_min(1e-6)[None, :, None]
            continuous = mask[:, 1:] & mask[:, :-1] & (dt[None] > 1e-5) & (dt[None] <= .5)
            continuous &= item['frame_indices'][None, 1:] != item['frame_indices'][None, :-1]
            velocity[:, 1:] *= continuous[:, :, None]
            relative = torch.cat([delta.abs(), delta.square().sum(-1, keepdim=True).sqrt(),
                                  velocity.abs().clamp_max(20), mask[:, :, None]], -1)
            relative = F.interpolate((relative * mask[:, :, None]).permute(0, 2, 1),
                                     size=target_t, mode='linear', align_corners=False)
            pair_features = torch.cat([(z[i] + z[j]) / 2, (z[i] - z[j]).abs(), relative], 1)
            pred = self.pair_head(pair_features).squeeze(1)
            # Average coverage preserves brief evidence between stride anchors.
            down = F.adaptive_avg_pool1d(mask[:, None].float(), target_t).squeeze(1)
            outputs.append((pred * down).sum(-1) / down.sum(-1).clamp_min(1))
        return torch.cat(outputs).max()


def interaction_rois(clip, source_shape, max_rois=4, padding=.2):
    """Rank box pairs by proximity and relative motion, without a pose gate.

    Return fixed union crops over the observed window. Padding is applied on
    each side as a fraction of the union's width/height and clipped to source.
    """
    h, w = _shape(source_shape)
    if max_rois < 0 or not math.isfinite(padding) or padding < 0:
        raise ValueError('Invalid ROI limits')
    _, boxes, times, indices = _clip_arrays(clip)
    first = _unique_mask(indices)
    valid = (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])
    ids = clip.get('track_ids', list(range(len(boxes))))
    if len(ids) != len(boxes):
        raise ValueError('track_ids must match people dimension')
    candidates = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            shared = valid[i] & valid[j] & first
            if shared.sum() < 4:
                continue
            a, b = boxes[i, shared], boxes[j, shared]
            scale = ((a[:, 2:] - a[:, :2]).norm(dim=-1) + (b[:, 2:] - b[:, :2]).norm(dim=-1)).median().clamp_min(1) / 2
            gap = torch.maximum(torch.maximum(a[:, :2] - b[:, 2:], b[:, :2] - a[:, 2:]), torch.zeros_like(a[:, :2]))
            distance = float(gap.norm(dim=-1).median() / scale)
            delta = ((a[:, :2] + a[:, 2:]) - (b[:, :2] + b[:, 2:])) / (2 * scale)
            shared_times = times[shared]
            dt = shared_times[1:] - shared_times[:-1]
            continuous = (dt > 1e-5) & (dt <= .5)
            speeds = ((delta[1:] - delta[:-1]).norm(dim=-1) / dt.clamp_min(1e-6))[continuous]
            motion = float(speeds.median()) if len(speeds) else 0.
            observations = torch.cat([boxes[i, valid[i]], boxes[j, valid[j]]])
            low, high = observations[:, :2].amin(0), observations[:, 2:].amax(0)
            pad = (high - low) * padding
            box = [max(0, math.floor(float(low[0] - pad[0]))), max(0, math.floor(float(low[1] - pad[1]))),
                   min(w, math.ceil(float(high[0] + pad[0]))), min(h, math.ceil(float(high[1] + pad[1])))]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            coverage = float(shared.sum() / max(1, int(first.sum())))
            rank = coverage / (1 + distance) + .25 * min(motion, 4.) / 4
            candidates.append(dict(box=box, track_ids=[ids[i], ids[j]], pair_indices=[i, j],
                                   quality=rank, shared_fraction=coverage))
    # Geometry tie-break keeps selection invariant to track-array permutation.
    candidates.sort(key=lambda x: (-x['quality'], tuple(x['box'])))
    selected = []
    for candidate in candidates:
        if any(candidate['box'] == previous['box'] for previous in selected):
            continue
        if len(selected) >= max_rois:
            break
        selected.append(candidate)
    return selected


def rgb_array(images):
    if not len(images):
        raise ValueError('At least one source frame required')
    result = []
    for image in images:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) <= 0:
            raise ValueError('RGB input requires nonempty uint8 BGR source frames')
        h, w = image.shape[:2]
        factor = 112 / max(h, w)
        small = cv2.resize(image, (max(1, round(w * factor)), max(1, round(h * factor))))
        canvas = np.zeros((112, 112, 3), dtype=np.uint8)
        y, x = (112 - small.shape[0]) // 2, (112 - small.shape[1]) // 2
        canvas[y:y + small.shape[0], x:x + small.shape[1]] = small[..., ::-1]
        result.append(canvas)
    return np.stack(result)


@lru_cache(maxsize=1)
def raw_pose_predictor_type():
    """Candidate-only adapter preserving coordinates before Results masks them.

    Ultralytics 8.2 Keypoints zeros xy below confidence .5 in place. Follow its
    PosePredictor NMS/scaling, then clone before wrapping. Standard Results and
    library globals retain their existing behavior for baseline consumers.
    """
    from ultralytics.models.yolo.pose.predict import PosePredictor
    from ultralytics.engine.results import Results
    from ultralytics.utils import ops

    class FusionRawPosePredictor(PosePredictor):
        def postprocess(self, preds, img, orig_imgs):
            preds = ops.non_max_suppression(
                preds, self.args.conf, self.args.iou,
                agnostic=self.args.agnostic_nms, max_det=self.args.max_det,
                classes=self.args.classes, nc=len(self.model.names))
            if not isinstance(orig_imgs, list):
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
            results = []
            for index, pred in enumerate(preds):
                original = orig_imgs[index]
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], original.shape).round()
                points = pred[:, 6:].reshape(len(pred), *self.model.kpt_shape)
                points = ops.scale_coords(img.shape[2:], points, original.shape)
                raw = points.clone()
                result = Results(original, path=self.batch[0][index], names=self.model.names,
                                 boxes=pred[:, :6], keypoints=points)
                result.fusion_raw_keypoints = raw
                results.append(result)
            return results

    return FusionRawPosePredictor


def ensure_raw_pose_predictor(pose_model):
    """Install on one YOLO instance, even if a preview initialized it earlier.

    YOLO.predict ignores predictor= after first use, so explicitly replace that
    instance's predictor. FakePose test doubles keep exposing their own tensors.
    """
    if not hasattr(pose_model, 'predictor') or not hasattr(pose_model, 'overrides'):
        return False
    from ultralytics.engine.model import Model
    if not isinstance(pose_model, Model):
        return False
    predictor_type = raw_pose_predictor_type()
    if isinstance(pose_model.predictor, predictor_type):
        return True
    overrides = dict(pose_model.overrides)
    overrides.update(task='pose', mode='predict', device=str(pose_model.device), save=False)
    predictor = predictor_type(overrides=overrides, _callbacks=pose_model.callbacks)
    predictor.setup_model(model=pose_model.model, verbose=False)
    pose_model.predictor = predictor
    return True


def extract_pose_clip(images_BGR, timestamps, frame_indices, pose_model,
                      camera_id='fight-fusion'):
    """Shared deterministic window-local detection/association for train/replay.

    Only actual observations are stored. Low-confidence boxes can recover an
    existing track but cannot create one, and duplicate source frames are never
    presented to the tracker as additional observations.
    """
    from .detector import Detection
    from .tracker import ByteTracker

    n = len(images_BGR)
    if n == 0 or len(timestamps) != n or len(frame_indices) != n:
        raise ValueError('Nonempty matching frames, timestamps and frame indices required')
    times = np.asarray(timestamps, dtype=np.float64)
    indices = np.asarray(frame_indices, dtype=np.int64)
    if (not np.isfinite(times).all() or np.any(np.diff(times) < 0)
            or np.any(np.diff(indices) < 0)):
        raise ValueError('Source observations must be finite and chronological')
    shape = images_BGR[0].shape[:2]
    if any(image.shape[:2] != shape for image in images_BGR):
        raise ValueError('Frame dimensions changed inside a window')
    positions = np.flatnonzero(np.r_[True, np.diff(indices) != 0]).tolist()
    for pos in range(1, n):
        if (indices[pos] == indices[pos - 1]) != (times[pos] == times[pos - 1]):
            raise ValueError('Repeated source frame must retain its original timestamp')
    tracker = ByteTracker(track_thresh=.25, low_thresh=.1, track_buffer=4,
                          min_hits=1, camera_id=camera_id)
    device = str(pose_model.device)
    expects_raw = ensure_raw_pose_predictor(pose_model)
    observed = {}
    for begin in range(0, len(positions), 4):
        batch_positions = positions[begin:begin + 4]
        predictions = pose_model.predict([images_BGR[p] for p in batch_positions],
                                         imgsz=640, conf=.1, iou=.45,
                                         half=False, device=device, verbose=False,
                                         max_det=100, save=False)
        if len(predictions) != len(batch_positions):
            raise ValueError('Pose result count does not match source frames')
        for position, result in zip(batch_positions, predictions):
            source_ts = float(times[position])
            ts = source_ts - float(times[0]) + 1e-9
            boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
            raw_points = getattr(result, 'fusion_raw_keypoints', None)
            if expects_raw and raw_points is None:
                raise ValueError('Candidate pose predictor omitted raw joint coordinates')
            if not len(boxes):
                joints = np.empty((0, 17, 3), np.float32)
            elif raw_points is not None:
                joints = raw_points.detach().cpu().numpy().astype(np.float32)
            else:
                joints = (result.keypoints.data.detach().cpu().numpy().astype(np.float32)
                          if result.keypoints is not None else np.empty((0, 17, 3), np.float32))
            if joints.shape != (len(boxes), 17, 3) or scores.shape != (len(boxes),):
                raise ValueError('Pose boxes and keypoints are not aligned')
            tracker._tracks = [track for track in tracker.all_tracks if ts - track.last_seen <= .5]
            detections = [Detection(box.tolist(), float(score), timestamp=ts)
                          for box, score in zip(boxes, scores)]
            tracks = tracker.update(detections)
            used, current = set(), []
            for track in tracks:
                if track.time_since_update:
                    continue
                for index, box in enumerate(boxes):
                    if index not in used and np.allclose(box, track.bbox, atol=1e-4, rtol=0):
                        current.append((track.track_id, joints[index], box))
                        used.add(index)
                        break
            observed[int(indices[position])] = current
    ids = sorted({tid for observations in observed.values() for tid, _, _ in observations})
    lookup = {tid: index for index, tid in enumerate(ids)}
    joints = np.zeros((len(ids), n, 17, 3), np.float32)
    boxes = np.zeros((len(ids), n, 4), np.float32)
    for position, index in enumerate(indices):
        for tid, points, box in observed[int(index)]:
            joints[lookup[tid], position] = points
            boxes[lookup[tid], position] = box
    return dict(keypoints=torch.from_numpy(joints), boxes=torch.from_numpy(boxes),
                frame_indices=indices.tolist(), timestamps=times.tolist(), track_ids=ids,
                start=float(times[0]), end=float(times[-1]))


def crop_rgb_frames(images_BGR, rois):
    if not len(images_BGR):
        raise ValueError('At least one source frame required')
    shape = images_BGR[0].shape[:2]
    if any(image.shape[:2] != shape for image in images_BGR):
        raise ValueError('Frame dimensions changed inside a window')
    h, w = shape
    result = []
    for roi in rois:
        x1, y1, x2, y2 = map(int, roi['box'])
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError('ROI lies outside source image')
        result.append(rgb_array([image[y1:y2, x1:x2] for image in images_BGR]))
    return result


def rgb_tensor(rgb):
    value = torch.as_tensor(rgb).float().permute(3, 0, 1, 2) / 255
    mean = torch.tensor([.43216, .394666, .37645])[:, None, None, None]
    std = torch.tensor([.22803, .22145, .216989])[:, None, None, None]
    return (value - mean) / std


def make_rgb_model(pretrained=True):
    from torchvision.models.video import r3d_18
    model = r3d_18(weights=None)
    if pretrained:
        if not KINETICS_PATH.is_file():
            raise FileNotFoundError(f'Local Kinetics checkpoint required: {KINETICS_PATH}')
        model.load_state_dict(torch.load(KINETICS_PATH, map_location='cpu', weights_only=True))
    model.fc = nn.Linear(512, 2)
    return model


def rgb_logit(model, rgb):
    """Training-compatible frozen-layer3 boundary; caller controls AMP/grad."""
    device = next(model.parameters()).device
    x = rgb_tensor(rgb).unsqueeze(0).to(device)
    z = model.layer3(model.layer2(model.layer1(model.stem(x))))
    z = model.avgpool(model.layer4(z.half().float() if device.type == 'cuda' else z)).flatten(1)
    logits = model.fc(z)
    return (logits[0, 1] - logits[0, 0]).float()


def window_quality(clip, source_shape, rois):
    h, w = _shape(source_shape)
    k, boxes, times, indices = _clip_arrays(clip)
    first = _unique_mask(indices)
    n = max(1, int(first.sum()))
    valid = (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])
    pose_valid = ((k[:, :, 5:, 2] >= .1).sum(-1) >= 2) & valid
    confidence = k[:, first, 5:, 2]
    confidence = confidence[confidence >= .01]
    heights = (boxes[..., 3] - boxes[..., 1])[valid & first[None]]
    continuous = valid[:, 1:] & valid[:, :-1] & first[None, 1:]
    if len(times) > 1:
        continuous &= ((times[1:] - times[:-1] > 1e-5) & (times[1:] - times[:-1] <= .5))[None]
    continuity = float(continuous.sum(-1).max() / max(1, n - 1)) if len(valid) else 0.
    values = dict(
        unique_frame_fraction=float(first.sum() / max(1, len(indices))),
        box_pair_coverage=float(((valid.sum(0) >= 2) & first).sum() / n),
        pose_pair_coverage=float(((pose_valid.sum(0) >= 2) & first).sum() / n),
        mean_joint_confidence=float(confidence.mean()) if confidence.numel() else 0.,
        person_height_ratio=float(heights.median() / h) if len(heights) else 0.,
        track_continuity=continuity,
        roi_area_fraction=max([(r['box'][2] - r['box'][0]) * (r['box'][3] - r['box'][1]) / (h * w) for r in rois], default=0.),
        roi_count_fraction=len(rois) / 4,
    )
    return {name: min(1., max(0., values[name])) for name in QUALITY_NAMES}


def fusion_features(logits, quality, available):
    """38 features: logits, masks, quality, then all quality/logit interactions."""
    if len(logits) != 3 or len(available) != 3:
        raise ValueError('Exactly three branch logits and availability masks required')
    if set(quality) != set(QUALITY_NAMES):
        raise ValueError('Quality fields do not match the sealed feature schema')
    scores = []
    for value, present in zip(logits, available):
        if present and (value is None or not math.isfinite(float(value))):
            raise ValueError('Available branch requires a finite logit')
        scores.append(float(value) if present else 0.)
    q = [float(quality[name]) for name in QUALITY_NAMES]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in q):
        raise ValueError('Quality must be finite within [0,1]')
    return torch.tensor(scores + [float(bool(a)) for a in available] + q
                        + [score * value for score in scores for value in q], dtype=torch.float32)


class QualityFusion(nn.Module):
    def __init__(self, missing_dropout=.2):
        super().__init__()
        if not 0 <= missing_dropout <= 1:
            raise ValueError('Invalid missing-modality dropout')
        self.missing_dropout = missing_dropout
        self.linear = nn.Linear(len(FUSION_FEATURE_NAMES), 1)
        self.register_buffer('feature_mean', torch.zeros(len(FUSION_FEATURE_NAMES)))
        self.register_buffer('feature_std', torch.ones(len(FUSION_FEATURE_NAMES)))

    @torch.no_grad()
    def set_normalization(self, features):
        features = torch.as_tensor(features, dtype=torch.float32, device=self.feature_mean.device)
        if features.ndim != 2 or features.shape[1] != len(FUSION_FEATURE_NAMES) or not len(features) or not torch.isfinite(features).all():
            raise ValueError('Finite fusion-fit features required for normalization')
        self.feature_mean.copy_(features.mean(0))
        std = features.std(0, unbiased=False)
        # Constant availability columns become variable under modality dropout;
        # division by epsilon would manufacture enormous missing-branch values.
        self.feature_std.copy_(torch.where(std >= 1e-3, std, torch.ones_like(std)))

    def forward(self, features, missing_dropout=None):
        if features.shape[-1] != len(FUSION_FEATURE_NAMES):
            raise ValueError('Fusion feature dimension mismatch')
        probability = self.missing_dropout if missing_dropout is None else missing_dropout
        if not 0 <= probability <= 1:
            raise ValueError('Invalid missing-modality dropout')
        x = features
        if self.training and probability:
            x = x.clone()
            for branch in (1, 2):
                keep = (torch.rand(x.shape[:-1], device=x.device) >= probability).to(x.dtype)
                x[..., branch] *= keep
                x[..., 3 + branch] *= keep
                start = 6 + len(QUALITY_NAMES) + branch * len(QUALITY_NAMES)
                x[..., start:start + len(QUALITY_NAMES)] *= keep[..., None]
        return self.linear((x - self.feature_mean) / self.feature_std).squeeze(-1)


def _load_verified(spec, model, device):
    path = Path(spec['path'])
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if digest != spec['sha256']:
        raise ValueError(f'Candidate checkpoint hash mismatch: {path}')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(model, QualityFusion) and 'feature_names' in saved and tuple(saved['feature_names']) != FUSION_FEATURE_NAMES:
        raise ValueError('Fusion checkpoint feature schema mismatch')
    model.load_state_dict(saved.get('state_dict', saved), strict=True)
    return model.to(device).eval()


class FightFusionPredictor:
    """Load sealed {global,roi,skeleton,fusion}:{path,sha256} checkpoints.

    Additional settings: threshold (required), device (default cuda), max_rois
    (default 4), padding (default .2). No inference path downloads weights.
    """
    def __init__(self, candidate_config):
        self.config = (json.loads(Path(candidate_config).read_text(encoding='utf-8'))
                       if isinstance(candidate_config, (str, Path)) else dict(candidate_config))
        self.device = torch.device(self.config.get('device', 'cuda'))
        self.threshold = float(self.config['threshold'])
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError('Invalid candidate threshold')
        self.global_model = _load_verified(self.config['global'], make_rgb_model(False), self.device)
        self.roi_model = _load_verified(self.config['roi'], make_rgb_model(False), self.device)
        self.skeleton_model = _load_verified(self.config['skeleton'], FusionSkeletonModel(), self.device)
        self.fusion_model = _load_verified(self.config['fusion'], QualityFusion(), self.device)

    @torch.inference_mode()
    def score_window(self, images_BGR, clip, source_shape):
        h, w = _shape(source_shape)
        if len(images_BGR) != len(clip['timestamps']) or not len(images_BGR):
            raise ValueError('RGB and pose must describe the same nonempty window')
        if any(image.shape[:2] != (h, w) for image in images_BGR):
            raise ValueError('RGB dimensions must match pose source dimensions')
        rois = interaction_rois(clip, source_shape, self.config.get('max_rois', 4), self.config.get('padding', .2))
        quality = window_quality(clip, source_shape, rois)
        item = {name: value.to(self.device) for name, value in prepare_fusion_skeleton(clip, source_shape).items()}
        context = torch.autocast('cuda') if self.device.type == 'cuda' else nullcontext()
        with context:
            full = float(rgb_logit(self.global_model, rgb_array(images_BGR)))
            local_scores = [float(rgb_logit(self.roi_model, rgb)) for rgb in crop_rgb_frames(images_BGR, rois)]
            local = max(local_scores) if local_scores else None
            skeleton = self.skeleton_model(item)
            skeleton = float(skeleton.float()) if skeleton is not None else None
        logits = [full, local, skeleton]
        availability = [value is not None for value in logits]
        features = fusion_features(logits, quality, availability).to(self.device)
        fused = self.fusion_model(features).float()
        if not torch.isfinite(fused):
            raise ValueError('Nonfinite fused logit')
        score = float(fused.sigmoid())
        return dict(score=score, logit=float(fused), is_fight=score >= self.threshold,
                    branch_logits=dict(zip(BRANCH_NAMES, logits)),
                    availability=dict(zip(BRANCH_NAMES, availability)), quality=quality, rois=rois)
