"""Compare fight evidence gates on existing validation caches and saved events.

No training, threshold search, event writes, or running-service changes. Saved
event clips are sparse/lossy and cannot reproduce the original live input.
"""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.update(YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false')

import cv2
import torch
from backend.vision.action_guards import FIGHT_PEOPLE_POLICY, fight_people_evidence, single_person_only
from backend.vision.live_actions import LiveActionModels, sample_live_frames


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def counts(rows, field, threshold):
    result = dict(tp=0, fp=0, tn=0, fn=0, unknown=0)
    for row in rows:
        score = row[field]
        positive = score is not None and score >= threshold
        result['tp' if row['label'] and positive else 'fn' if row['label'] else 'fp' if positive else 'tn'] += 1
        result['unknown'] += score is None
    return result


def main():
    output = ROOT / 'results/live_actions/fight_people_v3'
    output.mkdir(parents=True, exist_ok=True)
    dump(output / 'protocol.json', dict(policy=FIGHT_PEOPLE_POLICY, purpose=__doc__,
         limitation='Previously inspected validation set; not an independent final test. Unknown positive cases count as misses.'))
    models = LiveActionModels(ROOT / 'config/live_actions.json')
    source = ROOT / 'results/video_events/skeleton_comparison_round2/vfd/rgb/seed_42/finetune_validation_predictions.json'
    threshold = models.manifest['fight']['threshold']
    rows = []
    for index, entry in enumerate(json.loads(source.read_text())['rows']):
        cache = torch.load(ROOT / f'datasets/video_events/skeleton_rebuild/vfd/{entry["sample_id"]}.pt',
                           map_location='cpu', weights_only=True)
        clips = []
        for clip in cache['clips']:
            score = models.score_rgb_clip(clip['rgb'])
            people = fight_people_evidence(clip)
            clips.append(dict(score=score, previous=None if single_person_only(clip) else score,
                              revised=score if people['eligible'] else None, people=people))
        maximum = lambda key: max((c[key] for c in clips if c[key] is not None), default=None)
        rows.append(dict(sample_id=entry['sample_id'], label=entry['label'], raw=maximum('score'),
                         previous=maximum('previous'), revised=maximum('revised'), clips=clips))
        if (index + 1) % 25 == 0:
            print(f'validation {index + 1}/207', flush=True)
    summary = {key: counts(rows, key, threshold) for key in ('raw', 'previous', 'revised')}
    dump(output / 'validation.json', dict(threshold=threshold, summary=summary, rows=rows))
    print(json.dumps(summary), flush=True)

    cases = []
    events = json.loads((ROOT / 'results/live_actions/new_events_20260917/events.json').read_text(encoding='utf-8'))
    for event in events:
        visual = event['rule_basis']['visual']
        cap = cv2.VideoCapture(str(ROOT / 'data/events' / visual['evidence_clip']))
        images = []
        while True:
            ok, image = cap.read()
            if not ok:
                break
            images.append(image)
        cap.release()
        first, last = visual['frame_times'][0], visual['frame_times'][-1]
        frames = [(first + i * (last-first)/(len(images)-1), im, {}) for i, im in enumerate(images)]
        windows = []
        for end in range(8, len(frames)+1):
            source_frames = [f for f in frames[:end] if f[0] >= frames[end-1][0]-4]
            selected, reason = sample_live_frames(source_frames)
            if not selected:
                windows.append(dict(state='warming', reason=reason))
                continue
            clip = models.pose.clip(event['event_id'], source_frames, selected)
            people = fight_people_evidence(clip)
            windows.append(dict(end_offset=round(source_frames[-1][0]-first, 3),
                                previous_veto=single_person_only(clip), people=people,
                                full_result=models.predict(event['event_id'], source_frames)))
        cases.append(dict(event_id=event['event_id'], original_type=event['event_type'],
                          original_score=visual['score'], windows=windows))
        print(json.dumps(dict(event=event['event_id'], eligible_windows=sum(w.get('people',{}).get('eligible',False) for w in windows))), flush=True)
    dump(output / 'saved_events.json', cases)


if __name__ == '__main__':
    main()
