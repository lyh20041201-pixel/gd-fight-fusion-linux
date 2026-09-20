"""Shared ST-GCN++ A/B protocol, bounded stores, and exact stochastic max-MIL."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import copy, random, time, math
from collections import OrderedDict
import numpy as np
import torch
from scripts.skeleton_common import ROOT,CACHE,POSE,PROTOCOL,read,sha,digest,seal,offline
from scripts.skeleton_io import write
from scripts.skeleton_round2 import MANIFESTS,require_source_seal,choose_threshold,selection_key,from_confusion
from scripts.train_skeleton_round2 import SampleStore,tensor_bytes,rng_state,restore_rng,checkpoint
from scripts.train_skeleton_comparison import save_torch
from scripts.evaluate_skeleton_comparison import observation_quality
from backend.vision.stgcnpp_actions import STGCNPPActionModel,prepare_stgcnpp,WEIGHTS,WEIGHTS_SHA,BACKBONE_CFG

OUT=ROOT/'results/video_events/skeleton_stgcnpp_ab'
FEATURE_ROOT=CACHE/'stgcnpp_ab'
CODE=['scripts/stgcnpp_ab.py','scripts/train_stgcnpp_ab.py','backend/vision/stgcnpp_actions.py','backend/vision/stgcnpp_backbone.py',
      'scripts/skeleton_common.py','scripts/skeleton_round2.py','scripts/train_skeleton_round2.py','backend/vision/skeleton_actions.py']
POLICY=dict(version=1,seeds=[42,43,44],arms=['A','B'],datasets=['fallvision','vfd'],
    max_total_epochs=50,patience=8,optimizer='AdamW',learning_rate=.001,weight_decay=.0001,gradient_accumulation=4,gradient_clip=5.,
    class_weight='all training source negatives / all training source positives, unknowns included',
    architecture='pinned ST-GCN++ COCO17 2D Joint backbone; 256-channel fall head or symmetric 518->128->1 temporal pair head',
    input='same cached YOLO-Pose; x,y normalized by full image to [-1,1], confidence; joint mask >=0.3; original track/pair eligibility and geometry',
    pretrained_input_difference='upstream HRNet vs target YOLO; upstream confidence mask 0.01 vs target 0.3; upstream 100 timestamps vs target 32',
    initialization={'A':'random torch backbone including default BN buffers and anatomical graph prior','B':'all 690 NTU60 2D backbone tensors, including learned graph and BN buffers'},
    heads='random identical per paired seed; NTU60 60-class head excluded; all parameters trained from epoch 1 in both arms',
    numerical='CUDA fp16 autocast, float32 sigmoid, deterministic algorithms, TF32 off; same A/B; previous round used fp16 sigmoid',
    batchnorm='training running statistics retained; max-MIL replay restores per-window BN state and post-scan state without in-place buffer mutation',
    selection='validation max event recall at normal FPR<=5%, then lower FPR, then macro F1; higher threshold ties; earliest epoch ties',
    unknown='same input eligibility for both arms; retained in total denominators; positive unknown is missed event',
    regression='GMD and TNUE original splits only after selection; prior training/test history; TNUE provisional labels',
    test='existing second-round retained test, already seen in prior experiments; all 12 selections sealed before model access; never retune on test',
    network='only user-authorized pinned source/NTU60 weight download; all subsequent execution offline',
    baseline='previous round results secondary reference; A vs B is the initialization comparison')

def setup_runtime():
    offline();torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('Local CUDA required')

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def tensor_digest(state):
    import hashlib
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def make_model(task,arm,seed):
    seed_all(seed);model=STGCNPPActionModel(task)
    initial=tensor_digest(model.state_dict())
    load=model.load_ntu60() if arm=='B' else None
    heads={k:v for k,v in model.state_dict().items() if not k.startswith('backbone.')}
    return model,dict(random_initial_model_sha256=initial,head_initial_sha256=tensor_digest(heads),load=load)

def manifests():
    return {name:require_source_seal(MANIFESTS/f'{name}.json') for name in POLICY['datasets']}

def feature_signature():
    return digest(dict(version=1,pose=sha(POSE),sampling=PROTOCOL,
        prepare=sha(ROOT/'backend/vision/stgcnpp_actions.py'),old_prepare=sha(ROOT/'backend/vision/skeleton_actions.py'),
        coordinate_policy=POLICY['input']))

def eligible(item,task):
    valid=item['valid'];m=len(valid)
    if m<(2 if task=='fight' else 1):return False
    if task=='fall':return True
    pairs=torch.triu_indices(m,m,offset=1)
    return bool(((valid[pairs[0]] & valid[pairs[1]]).sum(-1)>=8).any())

class ABStore:
    def __init__(self,name):
        self.name=name;self.signature=feature_signature();self.folder=FEATURE_ROOT/self.signature[:16]/name
        self.lru=OrderedDict();self.bytes=0;self.max_bytes=128*1024*1024;self.max_items=8
    def path(self,row):return self.folder/(row['sample_id']+'.pt')
    def get(self,row):
        sid=row['sample_id']
        if sid in self.lru:self.lru.move_to_end(sid);return self.lru[sid][0]
        value=torch.load(self.path(row),map_location='cpu',weights_only=True)
        if value['signature']!=self.signature or value['source_sha256']!=row['sha256']:raise ValueError('Prepared input signature mismatch')
        size=tensor_bytes(value)
        if size<=self.max_bytes:
            while self.lru and (self.bytes+size>self.max_bytes or len(self.lru)>=self.max_items):
                _,(_,prior)=self.lru.popitem(last=False);self.bytes-=prior
            self.lru[sid]=(value,size);self.bytes+=size
        return value

def prepare_dataset(name,manifest):
    rawstore=SampleStore(name,manifest['rows'],'skeleton');store=ABStore(name);started=time.monotonic();records=[]
    task=manifest.get('task','fall' if name in ['fallvision','gmd'] else 'fight')
    for index,row in enumerate(manifest['rows']):
        if sha(row['path'])!=row['sha256']:raise ValueError('Source SHA mismatch: '+row['path'])
        rawpath=CACHE/name/(row['sample_id']+'.pt');rawsha=sha(rawpath) if rawpath.exists() else None
        if store.path(row).exists():
            value=store.get(row)
            if value['raw_cache_sha256']!=rawsha:raise ValueError('Underlying pose cache changed')
        else:
            raw=rawstore.raw(row);items=[];windows=[];reasons=[];maxtracks=0
            if raw is not None:
                for clip in raw['clips']:
                    item=prepare_stgcnpp(clip,raw['source_shape']);quality=observation_quality(clip,item,task)
                    usable=eligible(item,task);assert usable==(quality['skeleton_unknown_reason'] is None)
                    items.append(item);maxtracks=max(maxtracks,len(item['valid']))
                    windows.append(dict(start=float(clip['start']),end=float(clip['end']),quality=quality,usable=usable))
                    if quality['skeleton_unknown_reason']:reasons.append(quality['skeleton_unknown_reason'])
                quality=[c['quality'] for c in raw['clips']];shape=raw['source_shape']
                ratio=float(np.median([q['median_person_height']/max(1,shape[0]) for q in quality])) if quality else 0.
                coverage=float(np.mean([q['frame_coverage'] for q in quality])) if quality else 0.
                simultaneous=max([int((c['boxes'][...,2]>c['boxes'][...,0]).sum(0).max()) for c in raw['clips']] or [0])
            else:ratio=coverage=simultaneous=0;reasons=['source_extraction_failed']
            usable=any(w['usable'] for w in windows)
            value=dict(signature=store.signature,source_sha256=row['sha256'],raw_cache_sha256=rawsha,
                source_cache_signature=rawstore.expected[row['sample_id']],items=items,windows=windows,usable=usable,
                unknown_reasons=sorted(set(reasons)) if not usable else [],max_tracks=maxtracks,
                diagnostics=dict(person_height_ratio=ratio,pose_frame_coverage=coverage,
                    person_size='small' if ratio<.15 else 'medium' if ratio<.35 else 'large',
                    pose_visibility_proxy='low' if coverage<.5 else 'partial' if coverage<.9 else 'high',
                    simultaneous_detections=simultaneous,crowd='multiple' if simultaneous>=3 else 'zero_to_two'))
            save_torch(store.path(row),value)
        records.append(dict(sample_id=row['sample_id'],path=row['path'],source_sha256=row['sha256'],label=row['label'],split=row['split'],group=row['group'],
            raw_cache=str(rawpath),raw_cache_sha256=rawsha,prepared_cache=str(store.path(row)),prepared_cache_sha256=sha(store.path(row)),
            usable=value['usable'],unknown_reasons=value['unknown_reasons'],windows=len(value['items']),max_tracks=value['max_tracks']))
        if (index+1)%100==0 or index+1==len(manifest['rows']):
            progress=dict(phase='prepare_verified_pose_inputs',dataset=name,completed=index+1,total=len(manifest['rows']),seconds=time.monotonic()-started,models_completed=0,models_total=12,epoch=0)
            write(OUT/'progress.json',progress);print(progress,flush=True)
    info=dict(dataset=name,feature_signature=store.signature,rows=records,unknown_counts={s:{str(label):sum(r['split']==s and r['label']==label and not r['usable'] for r in records) for label in [0,1]} for s in ['train','validation','test']})
    seal(OUT/'audit'/f'inputs_{name}.json',info)
    return info

def bn_state(model):
    return {name:{k:v.detach().clone() for k,v in module._buffers.items() if v is not None} for name,module in model.named_modules() if isinstance(module,torch.nn.modules.batchnorm._BatchNorm)}

def set_bn_state(model,state):
    # Replace buffer objects, rather than copy_, so already built backward graphs
    # keep their original tensor version counters. This matters for tied maxima.
    for name,module in model.named_modules():
        if name in state:
            for key,value in state[name].items():setattr(module,key,value.clone())

def window_logit(model,item):return model({k:v.cuda(non_blocking=False) for k,v in item.items()})

def training_logit(model,data):
    if not data['usable']:return None
    items=data['items']
    if len(items)==1:return window_logit(model,items[0])
    winners=[];best=None
    # Do not let autocast cache no-grad parameter casts and reuse them in the
    # gradient replay; otherwise learnable graph/conv parameters lose gradients.
    with torch.no_grad(),torch.autocast('cuda',cache_enabled=False):
        for i,item in enumerate(items):
            before=(torch.get_rng_state(),torch.cuda.get_rng_state_all());bn=bn_state(model)
            value=window_logit(model,item)
            if value is None:continue
            score=float(value)
            if not math.isfinite(score):raise ValueError('Nonfinite MIL score')
            if best is None or score>best:best=score;winners=[(i,before,bn)]
            elif score==best:winners.append((i,before,bn))
    after=(torch.get_rng_state(),torch.cuda.get_rng_state_all());after_bn=bn_state(model)
    values=[]
    try:
        for i,before,bn in winners:
            torch.set_rng_state(before[0]);torch.cuda.set_rng_state_all(before[1]);set_bn_state(model,bn)
            values.append(window_logit(model,items[i]).float())
    finally:
        torch.set_rng_state(after[0]);torch.cuda.set_rng_state_all(after[1]);set_bn_state(model,after_bn)
    return torch.stack(values).mean() if values else None

def score_sample(model,data,details=False):
    values=[]
    with torch.inference_mode(),torch.autocast('cuda'):
        for item in data['items']:
            value=window_logit(model,item)
            score=float(value.float().sigmoid()) if value is not None else None
            if score is not None and not math.isfinite(score):raise ValueError('Nonfinite inference score')
            values.append(score)
    known=[(i,v) for i,v in enumerate(values) if v is not None]
    peak=max(known,key=lambda iv:iv[1]) if known else None
    score=peak[1] if peak else None
    if (score is not None)!=data['usable']:raise ValueError('Model changed input eligibility')
    return (score,values,peak[0] if peak else None) if details else score

def evaluate(model,rows,store,progress=None):
    model.eval();scores=[];started=time.monotonic()
    for i,row in enumerate(rows):
        scores.append(score_sample(model,store.get(row)))
        if progress is not None and ((i+1)%50==0 or i+1==len(rows)):
            write(progress['path'],dict(**progress['info'],phase='validation',completed=i+1,total=len(rows),seconds=time.monotonic()-started))
    return scores

def model_config(dataset,arm,seed):
    return dict(dataset=dataset,arm=arm,seed=seed,policy=POLICY,backbone=BACKBONE_CFG,
        manifest_sha256=sha(MANIFESTS/f'{dataset}.json'),seal_sha256=sha(MANIFESTS/f'{dataset}.seal.json'),
        pose_sha256=sha(POSE),weight_sha256=WEIGHTS_SHA,feature_signature=feature_signature(),sampling_protocol=PROTOCOL,
        code_hashes={f:sha(ROOT/f) for f in CODE})

def predict(score,threshold):return -1 if score is None else int(score>=threshold)

def metrics_for(rows,key):
    cm=np.zeros((2,3),dtype=np.int64)
    for row in rows:
        p=row['results'][key]['prediction'];cm[row['label'],2 if p<0 else p]+=1
    return from_confusion(cm)
