"""A separate, causal, paired binary vs fine-action ST-GCN++ pilot.

The classes describe evidence in the PAST four seconds, not an instantaneous
pose. A fall remains positive while its dynamic interval is in that history;
static lying without a recent fall is negative. No future frames are sampled.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import math
import random
from collections import Counter, defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scripts.prepare_fine_labels_bc import ROOT, OUT, CACHE, read, write, sha, seal, offline
from scripts.label_fine_actions_bc import CLASSES
from backend.vision.stgcnpp_actions import STGCNPPActionModel, BACKBONE_CFG, WEIGHTS, WEIGHTS_SHA
from backend.vision.stgcnpp_backbone import STGCN

POLICY=dict(version=1,seeds=[42,43,44],arms=['B','C'],max_epochs=35,patience=8,min_epochs=12,
    learning_rate=.0003,weight_decay=.0001,batch_size=16,gradient_clip=5,
    optimizer='AdamW',numerics='CUDA fp16 autocast; deterministic; TF32 off; CPU threads 3',
    input='Same dense 8 Hz YOLOv8n-pose, last 4 seconds / 32 timestamps, 0.5s decision cadence; startup >=3.5s',
    primary_track='single-actor pilot: most valid unique timestamps, then confidence; decision uses past observations only',
    eligibility='>=8 unique original timestamps with >=4 confident body joints; joint confidence >=0.3; no fabricated frames',
    feature='image-centred x,y in [-1,1] and confidence; same masked temporal pooling as existing ST-GCN++',
    initialization='Same NTU60 pretrained backbone per paired seed; binary logits matched exactly at initialization by splitting normal prior across seven classes',
    objective='B: fall/normal CE; C: 8-class CE; identical per-video and binary-class sample weights; no action-class oversampling',
    temporal_label='Any >=0.25s dynamic fall in past window -> fall; otherwise most recent >=0.25s normal transition; otherwise last-second majority posture',
    uncertain='A partly observed transition <0.25s or an uncertain span makes training label unknown; not dropped from event evaluation',
    selection='Subject 3 only: maximize source-timestamp-matched event recall with <=5% normal-video false positives; then fewer total false alert clusters; then binary NLL; earliest exact tie',
    events='one source event per Fall video (adjacent directional parts merged); match alert after source onset and within 5s; separate pre-onset/late alerts; 4s cooldown',
    reporting='Normal clip false positives plus alert clusters per total and ready normal hours; startup and pose unknown retained; source clip truth distinct from AI temporal/action truth',
    baseline='A is deployed fall checkpoint core at its frozen threshold under common primary-track replay; not an end-to-end live camera/guard comparison',
    test='No test model predictions until all six epoch/threshold selections sealed; GMD subject 4 and previous FallVision test have prior evaluation history, not a new blind benchmark',
    decision='Multiclass worth expanding if >=2 of 3 paired seeds have >=1 fewer normal test FP with no recall loss, or >=2 extra TP with no FP increase; external recall degradation >2pp rejects replacement',
    deployment='Exploratory pilot only: no live-model replacement without representative local continuous-video evidence',
    downloads='none; reuse existing source videos, pose and NTU60 weights')
CODE=['scripts/fine_actions_bc.py','scripts/train_fine_actions_bc.py','scripts/label_fine_actions_bc.py',
      'scripts/prepare_fine_labels_bc.py','backend/vision/stgcnpp_backbone.py','backend/vision/stgcnpp_actions.py']


def setup():
    offline();torch.set_num_threads(3)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def tensor_hash(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        h.update(name.encode());h.update(t.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def overlap(a,b,x,y):return max(0.,min(b,y)-max(a,x))


def window_label(row,start,end):
    observed=[(s,overlap(start,end,s['start'],s['end'])) for s in row['spans']]
    if any(s['label']=='uncertain' and n>0 for s,n in observed):return -1
    falls=sum(n for s,n in observed if s['label']=='fall')
    if falls>=.25-1e-8:return 0
    if falls>1e-8:return -1
    transition=[s for s,n in observed if s['label'] in CLASSES[1:5] and n>=.25-1e-8]
    if transition:return CLASSES.index(max(transition,key=lambda s:min(end,s['end']))['label'])
    if any(n>1e-8 and s['label'] in CLASSES[1:5] for s,n in observed):return -1
    durations={c:sum(overlap(max(start,end-1),end,s['start'],s['end']) for s in row['spans'] if s['label']==c) for c in CLASSES[5:]}
    return CLASSES.index(max(durations,key=durations.get)) if max(durations.values())>0 else -1


def causal_indices(times,end):
    """Only timestamps available by end; select 32 uniformly in observed history."""
    times=np.asarray(times)
    available=np.flatnonzero((times>=end-4-1e-8)&(times<=end+1e-8))
    if end-times[0]<3.5-1e-8:return None,'warmup'
    if len(available)<8:return None,'low_frame_rate'
    t=times[available]
    if t[-1]-t[0]<3.5-1e-8 or np.diff(t).max()>.5:return None,'interrupted_history'
    selected=available[np.abs(t[:,None]-np.linspace(t[0],t[-1],32)[None,:]).argmin(0)]
    assert times[selected].max()<=end+1e-8
    return selected,None


def primary_features(z,indices):
    k=z['keypoints'][:,indices].copy()
    valid=(k[...,2]>=.3)[:,:,5:].sum(-1)>=4
    counts=np.asarray([len(np.unique(z['frame_indices'][indices][v])) for v in valid])
    if not len(counts) or counts.max()<8:return None,None,dict(reason='pose_unknown',eligible_tracks=0)
    usable=np.flatnonzero(counts>=8)
    primary=max(usable,key=lambda i:(counts[i],float(k[i,...,2].mean())))
    h,w=z['shape'];points=k[primary];visible=points[...,2]>=.3
    points[...,0]=(points[...,0]-w/2)/(w/2);points[...,1]=(points[...,1]-h/2)/(h/2)
    points[~visible]=0
    return points.transpose(2,0,1),valid[primary],dict(reason=None,eligible_tracks=len(usable),
        primary_track=int(z['track_ids'][primary]),valid_unique_frames=int(counts[primary]),
        visible_frame_fraction=float(valid[primary].mean()))


def prepare_windows():
    labels=read(OUT/'fine_annotations.json'); all_windows=[];x=[];masks=[]
    for row in labels['videos']:
        path=CACHE/'pose'/f'{row["id"]}.npz';z=np.load(path,allow_pickle=False)
        assert str(z['source_sha256'])==row['sha256']
        times=z['times']; last=float(times[-1])
        endpoints=list(np.arange(.5,last,.5))+[last]
        for end in endpoints:
            indices,reason=causal_indices(times,end)
            window=dict(video_id=row['id'],split=row['split'],end=float(end),start=max(0.,float(end)-4),
                        feature_index=-1,label=window_label(row,max(0.,float(end)-4),float(end)),reason=reason)
            if indices is not None:
                feat,valid,quality=primary_features(z,indices);window.update(quality)
                window['sampled_start']=float(times[indices[0]]);window['sampled_end']=float(times[indices[-1]])
                window['sampled_frame_indices']=z['frame_indices'][indices].tolist()
                if feat is not None:
                    window['feature_index']=len(x);x.append(feat);masks.append(valid)
            all_windows.append(window)
    features=np.stack(x).astype(np.float32);valid=np.stack(masks)
    np.savez_compressed(CACHE/'windows.npz',features=features,valid=valid)
    meta=dict(windows=all_windows,annotations_sha256=sha(OUT/'fine_annotations.json'),
        features_sha256=sha(CACHE/'windows.npz'),
        by_split={s:dict(windows=sum(w['split']==s for w in all_windows),
            usable=sum(w['split']==s and w['feature_index']>=0 for w in all_windows),
            usable_labeled=Counter(CLASSES[w['label']] for w in all_windows if w['split']==s and w['feature_index']>=0 and w['label']>=0),
            unknown_reasons=Counter(w['reason'] for w in all_windows if w['split']==s and w['reason'])) for s in ['train','validation','test']})
    seal(CACHE/'windows.json',meta)
    print(meta['by_split'],flush=True)


class FineActionModel(nn.Module):
    def __init__(self,nclasses):
        super().__init__();self.backbone=STGCN(**BACKBONE_CFG);self.head=nn.Linear(256,nclasses)

    def forward(self,x,valid):
        z=self.backbone(x.permute(0,2,3,1).unsqueeze(1))[:,0].mean(-1)
        mask=F.interpolate(valid[:,None].float(),size=z.shape[-1],mode='nearest')
        pooled=(z*mask).sum(-1)/mask.sum(-1).clamp_min(1)
        return self.head(pooled)


def make_model(arm,seed):
    seed_all(seed)
    base=STGCNPPActionModel('fall');base.load_ntu60()
    binary_head=nn.Linear(256,2)
    model=FineActionModel(2 if arm=='B' else len(CLASSES));model.backbone.load_state_dict(base.backbone.state_dict())
    with torch.no_grad():
        model.head.weight[0]=binary_head.weight[0];model.head.bias[0]=binary_head.bias[0]
        model.head.weight[1:]=binary_head.weight[1]
        model.head.bias[1:]=binary_head.bias[1]-(math.log(7) if arm=='C' else 0)
    record=dict(backbone_sha256=tensor_hash(model.backbone.state_dict()),binary_reference_head_sha256=tensor_hash(binary_head.state_dict()),pretrained_sha256=WEIGHTS_SHA)
    return model,record


def make_baseline():
    spec=read(ROOT/'config/live_actions.json')['fall'];assert sha(spec['path'])==spec['sha256']
    checkpoint=torch.load(spec['path'],map_location='cpu',weights_only=True)
    model=FineActionModel(2);model.backbone.load_state_dict({k.removeprefix('backbone.'):v for k,v in checkpoint['state_dict'].items() if k.startswith('backbone.')})
    with torch.no_grad():
        model.head.weight.zero_();model.head.bias.zero_()
        model.head.weight[0]=checkpoint['state_dict']['fall_head.weight'][0]
        model.head.bias[0]=checkpoint['state_dict']['fall_head.bias'][0]
    return model,spec


def load_data(split):
    meta=read(CACHE/'windows.json');z=np.load(CACHE/'windows.npz',allow_pickle=False)
    assert sha(OUT/'fine_annotations.json')==meta['annotations_sha256']
    assert sha(CACHE/'windows.npz')==meta['features_sha256']
    windows=[w for w in meta['windows'] if w['split']==split]
    rows=[r for r in read(OUT/'fine_annotations.json')['videos'] if r['split']==split]
    return dict(windows=windows,videos=rows,x=torch.from_numpy(z['features']),valid=torch.from_numpy(z['valid']))


def predict(model,data):
    model.eval();usable=[w for w in data['windows'] if w['feature_index']>=0];prob=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for start in range(0,len(usable),64):
            idx=[w['feature_index'] for w in usable[start:start+64]]
            logits=model(data['x'][idx].cuda(),data['valid'][idx].cuda())
            prob.extend(logits.float().softmax(-1).cpu().tolist())
    it=iter(prob);out=[]
    for w in data['windows']:
        p=next(it) if w['feature_index']>=0 else None
        out.append(dict(**w,score=p[0] if p else None,probabilities=p))
    return out


def alert_times(windows,threshold):
    alerts=[]
    for w in windows:
        if w['score'] is not None and w['score']>=threshold and (not alerts or w['end']-alerts[-1]>=4):alerts.append(w['end'])
    return alerts


def metrics(videos,predictions,threshold):
    byid=defaultdict(list)
    for p in predictions:byid[p['video_id']].append(p)
    tp=fn=fp=tn=0;normal_alerts=unmatched_alerts=0;delays=[];details=[];unknown=0;normal_hours=ready_hours=0
    per_action=defaultdict(lambda:dict(videos=0,false_positive_videos=0))
    for row in videos:
        wins=byid[row['id']];alerts=alert_times(wins,threshold);fall=row['category']=='Fall'
        if not any(w['score'] is not None for w in wins):unknown+=1
        if fall:
            onset=row['events'][0]['start'];deadline=min(row['duration'],onset+5)
            matched=[t for t in alerts if onset<=t<=deadline]
            hit=bool(matched);tp+=int(hit);fn+=int(not hit)
            # One matched event, all additional clusters are spurious/duplicates.
            unmatched_alerts+=len(alerts)-int(hit)
            delay=matched[0]-onset if matched else None
            if delay is not None:delays.append(delay)
        else:
            hit=bool(alerts);fp+=int(hit);tn+=int(not hit);normal_alerts+=len(alerts);normal_hours+=row['duration']/3600
            previous=0
            for w in wins:
                if w['score'] is not None:ready_hours+=(w['end']-previous)/3600
                previous=w['end']
            actions=sorted({s['label'] for s in row['spans']})
            for action in actions:
                per_action[action]['videos']+=1;per_action[action]['false_positive_videos']+=int(hit)
            delay=None
        details.append(dict(id=row['id'],truth='fall' if fall else 'normal',alerts=alerts,detected=hit,delay_seconds=delay,
            usable_windows=sum(w['score'] is not None for w in wins),total_windows=len(wins)))
    precision=tp/(tp+normal_alerts+unmatched_alerts) if tp+normal_alerts+unmatched_alerts else 0
    return dict(tp=tp,fn=fn,fp=fp,tn=tn,event_recall=tp/max(1,tp+fn),normal_video_fpr=fp/max(1,fp+tn),
        event_precision=precision,normal_alerts=normal_alerts,other_false_or_duplicate_alerts=unmatched_alerts,
        false_alerts_total=normal_alerts+unmatched_alerts,normal_video_seconds=normal_hours*3600,normal_ready_seconds=ready_hours*3600,
        false_alerts_per_normal_hour=normal_alerts/normal_hours if normal_hours else None,
        false_alerts_per_ready_normal_hour=normal_alerts/ready_hours if ready_hours else None,
        median_detected_delay_seconds=float(np.median(delays)) if delays else None,
        p90_detected_delay_seconds=float(np.percentile(delays,90)) if delays else None,
        all_unknown_videos=unknown,unknown_window_fraction=sum(w['score'] is None for w in predictions)/max(1,len(predictions)),
        unknown_reasons=dict(Counter(w['reason'] for w in predictions if w['score'] is None)),
        per_action_normal_videos=dict(per_action),videos=details)


def binary_nll(predictions):
    losses=[]
    for w in predictions:
        if w['score'] is None or w['label']<0:continue
        p=max(1e-7,min(1-1e-7,w['score']));losses.append(-math.log(p if w['label']==0 else 1-p))
    return float(np.mean(losses)) if losses else 1e6


def select_threshold(videos,predictions):
    candidates=sorted({1.000001,*[float(w['score']) for w in predictions if w['score'] is not None]})
    best=None;nll=binary_nll(predictions)
    for threshold in candidates:
        m=metrics(videos,predictions,threshold)
        if m['normal_video_fpr']>.05+1e-8:continue
        key=(m['tp'],-m['false_alerts_total'],-nll,threshold)
        if best is None or key>best[0]:best=(key,threshold,m)
    assert best is not None
    return best[1],best[2],(best[0][0],best[0][1],best[0][2])


def save_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def freeze():
    labels=read(OUT/'fine_annotations_draft.json')
    labels['second_review']=dict(method='Same AI reviewer; 24 denser RGB clips; not independent human review',
        evidence={p.name:sha(p) for p in (OUT/'review').glob('boundary_*.jpg')},
        corrections=['Refined s2 ADL 03/17/20 transitions','Retained seated label for s3 ADL 20 after ambiguity check',
                     'Refined s1 ADL 04, s3 ADL 08/09, s4 ADL 14/16 boundaries','Merged s2 fall 10 directional parts; dynamic end 4.3s'])
    seal(OUT/'fine_annotations.json',labels)
    plan=dict(policy=POLICY,classes=CLASSES,annotations_sha256=sha(OUT/'fine_annotations.json'),
        sources_sha256=sha(OUT/'source_inventory.json'),pose_protocol_sha256=sha(CACHE/'pose_protocol.json'),
        code_sha256={p:sha(ROOT/p) for p in CODE},baseline=read(ROOT/'config/live_actions.json')['fall'])
    seal(OUT/'protocol_frozen.json',plan)
    prepare_windows()


def verify_seal():
    plan=read(OUT/'protocol_frozen.json')
    assert plan['policy']==POLICY
    assert plan['annotations_sha256']==sha(OUT/'fine_annotations.json')
    assert plan['code_sha256']=={p:sha(ROOT/p) for p in CODE},'Sealed training code changed'
    return plan
