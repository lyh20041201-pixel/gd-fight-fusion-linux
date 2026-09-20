"""Offline GMD source inventory, RGB review sheets and dense 8 Hz pose cache.

This experiment lives apart from previously sealed training artifacts. Source
video labels are retained verbatim; AI interval labels are a separate artifact.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import math
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scripts.skeleton_common import ROOT, POSE, offline, read, sha, seal
from scripts.skeleton_io import write

OUT = ROOT / 'results/live_actions/fine_labels_bc_v1'
CACHE = ROOT / 'datasets/video_events/fine_labels_bc_v1'


def inventory():
    data = read(ROOT / 'datasets/video_events/rebuild/gmd_audit.json')
    assert not data['errors']
    rows = []
    for row in data['videos']:
        row = dict(row)
        row['id'] = f"s{row['group'][-1]}_{row['category'].lower()}_{Path(row['path']).stem}"
        assert sha(row['path']) == row['sha256'], row['path']
        rows.append(row)
    assert len(rows) == len({r['sha256'] for r in rows}) == 160
    for a in rows:
        assert a['split'] == {'1':'train', '2':'train', '3':'validation', '4':'test'}[a['group'][-1]]
    result = dict(dataset='GMDCSA24', videos=rows,
                  source_annotation_sha256={str(p.relative_to(ROOT)):sha(p)
                    for p in (ROOT/'datasets/gmdcsa24').glob('Subject */*.csv')},
                  source_audit_sha256=sha(ROOT/'datasets/video_events/rebuild/gmd_audit.json'),
                  test_note='Subject 4 is excluded from this training/selection; dataset has prior experiment history, not newly blind data.')
    seal(OUT / 'source_inventory.json', result)
    return rows


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/consola.ttf', size)


def read_indices(path, indices):
    cap = cv2.VideoCapture(str(path)); frames = {}; wanted = set(indices)
    for idx in range(max(indices)+1):
        ok, frame = cap.read()
        if not ok: raise ValueError(f'Decode failed {path}: {idx}')
        if idx in wanted: frames[idx] = frame
    cap.release()
    return frames


def sheets(rows):
    folder = OUT / 'review'; folder.mkdir(parents=True, exist_ok=True)
    for subject in range(1, 5):
        for category in ['ADL', 'Fall']:
            subset = [r for r in rows if r['group'] == f'subject-{subject}' and r['category'] == category]
            for page in range(math.ceil(len(subset)/6)):
                batch = subset[page*6:(page+1)*6]
                sheet = Image.new('RGB', (1680, len(batch)*150+35), '#e9edf1')
                draw = ImageDraw.Draw(sheet)
                draw.text((5,5), f'RGB SOURCE REVIEW | subject {subject} | {category} | page {page+1} | AI review, NOT human truth', fill='black', font=font(19))
                evidence = []
                for j, row in enumerate(batch):
                    indices = np.linspace(0, row['frames']-1, 10).astype(int).tolist()
                    frames = read_indices(row['path'], indices)
                    y = 35+j*150
                    draw.text((3,y), f"{row['id']}  {row['duration']:.2f}s  {row['split']}", fill='black', font=font(16))
                    for i, idx in enumerate(indices):
                        im = Image.fromarray(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB))
                        im.thumbnail((166,110)); x = i*168
                        sheet.paste(im, (x, y+22))
                        draw.text((x+3,y+132), f'{idx/row["fps"]:.2f}s', fill='black', font=font(15))
                    evidence.append(dict(id=row['id'], frame_indices=indices, timestamps=[i/row['fps'] for i in indices]))
                name = f's{subject}_{category.lower()}_{page+1:02d}'
                sheet.save(folder/f'{name}.jpg', quality=92)
                write(folder/f'{name}.json', evidence)
                print('SHEET', name, flush=True)


def poses(rows):
    offline()
    import torch
    from ultralytics import YOLO
    from backend.vision.detector import Detection
    from backend.vision.tracker import ByteTracker
    torch.set_num_threads(3); cv2.setNumThreads(2)
    model = YOLO(str(POSE), task='pose')
    folder = CACHE/'pose'; folder.mkdir(parents=True, exist_ok=True)
    signature = dict(pose_sha256=sha(POSE), fps=8, imgsz=640, conf=.25, iou=.45,
                     joint_conf=.3, min_body_joints=4, tracking='production ByteTracker; expire after 0.5s',
                     tracker_sha256=sha(ROOT/'backend/vision/tracker.py'))
    seal(CACHE/'pose_protocol.json', signature)
    records = []
    started = time.monotonic()
    for row in rows:
        path = folder/f'{row["id"]}.npz'
        if path.exists():
            z=np.load(path, allow_pickle=False)
            assert str(z['source_sha256']) == row['sha256']
            records.append(dict(id=row['id'], path=str(path), frames=len(z['times']), tracks=len(z['track_ids'])))
            continue
        indices = np.unique(np.minimum(row['frames']-1, np.floor(np.arange(0, row['duration'], 1/8)*row['fps']).astype(int)))
        images = read_indices(row['path'], indices.tolist())
        tracker = ByteTracker(track_thresh=.25, low_thresh=.1, track_buffer=4, min_hits=1, camera_id=row['id'])
        observations = []; track_ids=set(); times = indices/row['fps']
        shape = images[int(indices[0])].shape[:2]
        for begin in range(0, len(indices), 8):
            batch=indices[begin:begin+8]
            results=model.predict([images[int(i)] for i in batch], imgsz=640, conf=.25, iou=.45,
                                  device=0, half=False, verbose=False, max_det=100, save=False)
            for frame_id,result in zip(batch,results):
                ts=float(frame_id/row['fps'])
                points=result.keypoints.data.cpu().numpy().astype(np.float32)
                boxes=result.boxes.xyxy.cpu().numpy().astype(np.float32)
                scores=result.boxes.conf.cpu().numpy().astype(np.float32)
                tracker._tracks=[t for t in tracker.all_tracks if ts-t.last_seen<=.5]
                tracks=tracker.update([Detection(b.tolist(),float(s),timestamp=ts) for b,s in zip(boxes,scores)])
                observed=[]; used=set()
                for track in tracks:
                    if track.time_since_update: continue
                    for k,box in enumerate(boxes):
                        if k not in used and np.allclose(box,track.bbox,atol=1e-4):
                            observed.append((track.track_id,points[k],box)); used.add(k);track_ids.add(track.track_id);break
                observations.append(observed)
        ids=sorted(track_ids); lookup={v:i for i,v in enumerate(ids)}
        points=np.zeros((len(ids),len(indices),17,3),np.float32)
        boxes=np.zeros((len(ids),len(indices),4),np.float32)
        for t,observed in enumerate(observations):
            for tid,p,b in observed: points[lookup[tid],t]=p; boxes[lookup[tid],t]=b
        np.savez_compressed(path, keypoints=points, boxes=boxes, times=times, frame_indices=indices,
                            track_ids=np.asarray(ids,dtype=np.int64), shape=np.asarray(shape), source_sha256=row['sha256'])
        records.append(dict(id=row['id'],path=str(path),frames=len(times),tracks=len(ids)))
        print('POSE',len(records),'/160',row['id'],'tracks',len(ids),'seconds',round(time.monotonic()-started,1),flush=True)
    write(CACHE/'pose_complete.json',dict(protocol=signature,rows=records,seconds=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['sheets','poses','inventory']);args=p.parse_args()
    rows=inventory()
    if args.command=='sheets':sheets(rows)
    if args.command=='poses':poses(rows)
