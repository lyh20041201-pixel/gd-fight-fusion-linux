"""Opt-in candidate adapter. Importing this module never changes live settings."""
from __future__ import annotations
from pathlib import Path
import hashlib
import time

import numpy as np

from .fight_fusion import FightFusionPredictor, extract_pose_clip
from .live_actions import sample_live_frames
from .action_guards import assess_view, GUARD_VERSION


class FightFusionLiveAdapter:
    def __init__(self, manifest, *, predictor=None, pose_model=None):
        self.predictor = predictor or FightFusionPredictor(manifest)
        self.config = self.predictor.config
        self.version = (hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
                        if isinstance(manifest, (str, Path)) else 'injected-test-candidate')
        if pose_model is None:
            from ultralytics import YOLO
            spec = self.config['pose']
            if hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest() != spec['sha256']:
                raise ValueError('Pose model hash mismatch')
            pose_model = YOLO(spec['path'], task='pose')
            pose_model.to(str(self.predictor.device))
        self.pose = pose_model

    def pose_preview(self, frame):
        result = self.pose.predict([frame[1]], imgsz=640, conf=.1, iou=.45,
                                   verbose=False, half=False, save=False, max_det=100)[0]
        return dict(timestamp=frame[0], keypoints=result.keypoints.data.cpu().numpy().tolist())

    def predict(self, camera, frames):
        started = time.monotonic()
        selected, reason = sample_live_frames(frames)
        if not selected:
            return dict(state='warming', reason=reason, actions=[], inference_ms=0.)
        shape = selected[0][1].shape[:2]
        if any(f[1].shape[:2] != shape for f in frames):
            return dict(state='warming', reason='画面尺寸变化，等待新窗口', actions=[], inference_ms=0.)
        view = assess_view(selected)
        if not view['ok']:
            return dict(state='blocked', reason=view['reason'], actions=[], quality=view,
                        inference_ms=round((time.monotonic()-started)*1000, 1), guard_version=GUARD_VERSION)
        images = [f[1] for f in selected]
        times = [f[0] for f in selected]
        unique = {t:i for i,t in enumerate(sorted(set(times)))}
        indices = [unique[t] for t in times]
        clip = extract_pose_clip(images, times, indices, self.pose, camera_id=str(camera))
        result = self.predictor.score_window(images, clip, shape)
        action = dict(kind='fight', label='打架', event_type='suspected_fight',
            threshold=self.predictor.threshold, model=self.config.get('label', '三路融合·候选'),
            score=result['score'], state='candidate' if result['is_fight'] else 'normal', reason=None,
            start=times[0], end=times[-1], model_version=self.version, guard_version=GUARD_VERSION,
            branch_logits=result['branch_logits'], availability=result['availability'],
            input_quality=result['quality'], interaction_regions=result['rois'])
        return dict(state='running', actions=[action],
                    inference_ms=round((time.monotonic()-started)*1000, 1),
                    unique_frames=len(unique), window_start=times[0], window_end=times[-1],
                    quality=view, guard_version=GUARD_VERSION, candidate_only=True)

