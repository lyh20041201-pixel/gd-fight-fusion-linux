"""Audit live input guards without modifying weights, live events or sealed results.

Cached validation evaluates only the semantic checks; it does not measure live
camera-motion detection or field accuracy. Saved event videos are sparse and
lossy, so their replay cannot exactly reproduce the original inference stream.
"""
from pathlib import Path
import argparse
import json
import os
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.update(YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false')

import cv2
import torch
from backend.vision.live_actions import LiveActionModels
from backend.vision.action_guards import GUARD_VERSION, single_person_only, rising_only_tracks, exclude_tracks


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def metrics(rows, key, threshold):
    counts = dict(tp=0, fp=0, tn=0, fn=0, unknown=0)
    for row in rows:
        value = row[key]
        positive = value is not None and value >= threshold
        counts['tp' if row['label'] and positive else 'fn' if row['label'] else 'fp' if positive else 'tn'] += 1
        counts['unknown'] += value is None
    return counts


def validation(models, output):
    summaries = {}
    datasets = [
        ('fallvision', 'fall', 'results/video_events/skeleton_stgcnpp_ab/fallvision/B/seed_42/best_validation_predictions.json'),
        ('vfd', 'fight', 'results/video_events/skeleton_comparison_round2/vfd/rgb/seed_42/finetune_validation_predictions.json')]
    for dataset, kind, path in datasets:
        rows = json.loads((ROOT / path).read_text())['rows']
        results = []
        for index, row in enumerate(rows):
            cache = torch.load(ROOT / f'datasets/video_events/skeleton_rebuild/{dataset}/{row["sample_id"]}.pt',
                               map_location='cpu', weights_only=True)
            baseline, guarded = [], []
            for clip in cache['clips']:
                if kind == 'fall':
                    raw = models.score_fall_clip(clip, cache['source_shape'])
                    excluded = rising_only_tracks(clip)
                    score = models.score_fall_clip(exclude_tracks(clip, excluded), cache['source_shape']) if any(excluded) else raw
                else:
                    raw = models.score_rgb_clip(clip['rgb'])
                    score = None if single_person_only(clip) else raw
                baseline.append(raw)
                guarded.append(score)
            aggregate = lambda values: max((v for v in values if v is not None), default=None)
            results.append(dict(sample_id=row['sample_id'], label=row['label'],
                                baseline=aggregate(baseline), guarded=aggregate(guarded)))
            if (index + 1) % 50 == 0:
                print(dataset, index + 1, '/', len(rows), flush=True)
        threshold = models.manifest[kind]['threshold']
        summary = dict(samples=len(results), threshold=threshold,
                       baseline=metrics(results, 'baseline', threshold), guarded=metrics(results, 'guarded', threshold))
        dump(output / (dataset + '_validation.json'), dict(summary=summary, rows=results))
        summaries[kind] = summary
        print(json.dumps({kind: summary}), flush=True)
    return summaries


def evidence(models, output):
    cases = []
    ids = ['VIS-daa0e8a495da460aa50db24ce44384e9', 'VIS-01d14953a1c9499ca57ddd35e675a927']
    with sqlite3.connect((ROOT / 'data/database/classroom.db').as_uri() + '?mode=ro', uri=True) as db:
        for eid in ids:
            row = db.execute('SELECT rule_basis, event_type FROM risk_events WHERE event_id=?', (eid,)).fetchone()
            if not row:
                continue
            basis = json.loads(row[0])['visual']
            cap = cv2.VideoCapture(str(ROOT / 'data/events' / basis['evidence_clip']))
            images = []
            try:
                while True:
                    ok, im = cap.read()
                    if not ok:
                        break
                    images.append(im)
            finally:
                cap.release()
            start, end = basis['frame_times'][0], basis['frame_times'][-1]
            frames = [(start + i * (end - start) / (len(images) - 1), im, {}) for i, im in enumerate(images)]
            windows = []
            for stop in range(9, len(frames) + 1):
                end_time = frames[stop - 1][0]
                window = [f for f in frames[:stop] if f[0] >= end_time - 4]
                result = models.predict(eid, window)
                windows.append(result)
            cases.append(dict(event_id=eid, original_type=row[1], original_score=basis['score'],
                              reconstructed_frames=len(frames), windows=windows))
    dump(output / 'saved_evidence.json', cases)
    return cases


def positive_replay(models, output):
    sample = 'f79cd02858173d60254e4d43'
    cache = torch.load(ROOT / f'datasets/video_events/skeleton_rebuild/fallvision/{sample}.pt', map_location='cpu', weights_only=True)
    path = ROOT / 'datasets/fallvision/raw/Fall Detection Video Dataset/Fall/Chair/Raw Video/f_raw_c_2/f_raw_c_2/C_N_78_resized.mp4'
    cap = cv2.VideoCapture(str(path))
    results = []
    try:
        for i, clip in enumerate(cache['clips']):
            frames = []
            for index, ts in zip(clip['frame_indices'], clip['timestamps']):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, im = cap.read()
                if not ok:
                    raise RuntimeError('Cannot decode positive replay')
                frames.append((float(ts), im, {}))
            results.append(models.predict('positive-' + str(i), frames))
    finally:
        cap.release()
    dump(output / 'positive_replay.json', results)
    return dict(sample_id=sample, windows=len(results), fall_candidates=sum(
        any(a['kind'] == 'fall' and a['state'] == 'candidate' for a in r['actions']) for r in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    models = LiveActionModels(ROOT / 'config/live_actions.json')
    report = dict(guard_version=GUARD_VERSION, limitations=__doc__)
    evidence(models, output)
    report['positive_replay'] = positive_replay(models, output)
    print(json.dumps(report['positive_replay']), flush=True)
    if args.validation:
        report['validation'] = validation(models, output)
    dump(output / 'report.json', report)


if __name__ == '__main__':
    main()
