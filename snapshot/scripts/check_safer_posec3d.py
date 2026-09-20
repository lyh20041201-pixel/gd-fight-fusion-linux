"""Bounded local replay using public weights and the already cached test poses."""
from pathlib import Path
import sys
import os
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
import time
import numpy as np
import torch
from backend.vision.posec3d_actions import load_posec3d, score_posec3d
from backend.vision.action_guards import rising_only_tracks, exclude_tracks
from scripts.fine_actions_bc import metrics

OUT = ROOT / 'results/live_actions/safer_posec3d_trial'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    model = load_posec3d(ROOT / 'models/safer_posec3d/non-wheelchair-epoch_44.pth')
    videos = json.loads((ROOT / 'results/live_actions/fine_labels_bc_v1/fine_annotations.json').read_text(encoding='utf-8'))['videos']
    videos = [v for v in videos if v['split'] == 'test']
    predictions = []
    elapsed = []
    for row in videos:
        z = np.load(ROOT / f'datasets/video_events/fine_labels_bc_v1/pose/{row["id"]}.npz', allow_pickle=False)
        times = z['times']
        # Fixed 2-second live cadence after 4-second warmup; include last frame
        # for a short-clip smoke test (not a continuous-camera benchmark).
        endpoints = sorted(set([*np.arange(4., float(times[-1]), 2.), float(times[-1])]))
        for end in endpoints:
            idx = np.flatnonzero((times >= end-4-1e-8) & (times <= end+1e-8))
            prediction = dict(video_id=row['id'], end=end, reason=None, score=None)
            if end < 3.5:
                prediction['reason'] = 'warmup'
            else:
                clip = dict(keypoints=torch.from_numpy(z['keypoints'][:, idx]),
                    boxes=torch.from_numpy(z['boxes'][:, idx]), frame_indices=z['frame_indices'][idx],
                    timestamps=times[idx] - times[idx][0], track_ids=z['track_ids'].tolist())
                clip = exclude_tracks(clip, rising_only_tracks(clip))
                started = time.perf_counter()
                result = score_posec3d(model, clip, z['shape'])
                elapsed.append(time.perf_counter()-started)
                prediction['raw'] = result
                prediction['score'] = result['score'] if result['is_fall'] else (0. if result['score'] is not None else None)
                if prediction['score'] is None:
                    prediction['reason'] = 'pose_unknown_or_rising'
            predictions.append(prediction)
        print(row['id'], len(predictions), flush=True)
    summary = metrics(videos, predictions, .3)
    output = dict(protocol='Public checkpoint unchanged; author demo threshold 0.3 plus fall argmax; existing 8Hz YOLOv8n pose; past-only timestamp resampling; rising guard; no RGB camera-motion guard in cached replay; two-second cadence plus clip endpoint.',
        limitations='Adaptation smoke test, not paper reproduction or directly comparable to previous 0.5-second primary-track experiment. No threshold tuning or retraining.',
        summary=summary, predictions=predictions,
        timing_seconds=dict(median=float(np.median(elapsed)), max=max(elapsed)))
    (OUT / 'replay.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k!='predictions'},ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
