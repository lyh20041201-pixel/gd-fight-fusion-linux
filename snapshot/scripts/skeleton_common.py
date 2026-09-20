"""Shared, offline-only protocol for the skeleton/RGB comparison."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'datasets/video_events/skeleton_rebuild'
OUT = ROOT / 'results/video_events/skeleton_comparison'
SOURCE = ROOT / 'datasets/video_events/rebuild'
POSE = ROOT / 'models/yolov8n-pose.pt'
POSE_URL = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt'
POSE_SIZE = 6832633
PROTOCOL = dict(version=1, window_seconds=4., frames=32, stride_seconds=2.,
                pose_imgsz=640, pose_conf=.25, joint_conf=.3,
                min_body_joints=4, min_valid_frames=8, max_gap_seconds=.5,
                seeds=[42,43,44], epochs=50, patience=8,
                aggregation='maximum instance then maximum window score',
                network_policy='Only explicitly authorized yolov8n-pose.pt; all execution offline')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def row_id(row):
    return digest([row['sha256'],row.get('start',0),row.get('end',row.get('duration'))])[:24]

def offline():
    os.environ.update(YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false',
                      HF_HUB_OFFLINE='1', WANDB_DISABLED='true')
    # Fail closed even if a dependency attempts a silent metadata/model request.
    import socket
    def denied(*args, **kwargs):
        raise RuntimeError('Network disabled for the offline comparison')
    socket.create_connection = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied

def seal(path, value):
    path = Path(path)
    if path.exists() and read(path) != value:
        raise ValueError(f'Sealed artifact differs: {path}; use a new experiment version')
    if not path.exists(): write(path,value)

def windows(row):
    start=float(row.get('start',0)); end=float(row.get('end',row.get('duration',0)))
    if end <= start: raise ValueError('Invalid interval')
    if end-start <=4: return [(start,end)]
    starts=list(np.arange(start,end-4,2.)); starts.append(end-4)
    return [(float(s),float(s+4)) for s in starts]

def frame_indices(start,end,fps,n):
    return np.linspace(round(start*fps),max(round(start*fps),min(n-1,round(end*fps)-1)),32).astype(np.int64)

def isolation(rows):
    if len({r['sample_id'] for r in rows}) != len(rows): raise ValueError('Duplicate sample ids')
    splits=sorted({r['split'] for r in rows})
    for i,a in enumerate(splits):
        for b in splits[i+1:]:
            for key in ('group','sha256'):
                if {r[key] for r in rows if r['split']==a} & {r[key] for r in rows if r['split']==b}:
                    raise ValueError(f'{key} leakage between {a} and {b}')

def metrics(truth,pred):
    """Unknown is a third prediction column and a false negative for its true class."""
    cm=np.zeros((2,3),dtype=int)
    for t,p in zip(truth,pred): cm[int(t),2 if p<0 else int(p)]+=1
    tp=np.array([cm[0,0],cm[1,1]],float)
    precision=tp/np.maximum(1,cm[:,:2].sum(0))
    recall=tp/np.maximum(1,cm.sum(1))
    f1=2*precision*recall/np.maximum(1e-12,precision+recall)
    return dict(samples=int(cm.sum()),confusion_matrix=cm.tolist(),
                prediction_columns=['normal','event','unknown'],precision=precision.tolist(),
                recall=recall.tolist(),f1=f1.tolist(),macro_f1=float(f1.mean()),
                accuracy=float(tp.sum()/max(1,cm.sum())),
                coverage=float(cm[:,:2].sum()/max(1,cm.sum())),
                normal_false_positive_rate=float(cm[0,1]/max(1,cm[0].sum())))

def choose_threshold(truth,scores):
    def predict(th): return [-1 if p is None else int(p>=th) for p in scores]
    candidates=[round(float(t),2) for t in np.arange(.05,.951,.01)]
    threshold=max(candidates,key=lambda t:(metrics(truth,predict(t))['macro_f1'],-abs(t-.5),-t))
    return threshold,metrics(truth,predict(threshold))

def rgb_tensor(rgb):
    import torch
    x=torch.as_tensor(rgb).float().permute(3,0,1,2)/255
    return (x-torch.tensor([.43216,.394666,.37645])[:,None,None,None])/torch.tensor([.22803,.22145,.216989])[:,None,None,None]
