"""Offline, bounded-memory, resumable round-two MIL training; validation only.

First-round entry points and outputs are intentionally untouched. All inference
and MIL scoring used by the round-two evaluator are exported from this module.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,copy,random,time,msvcrt
from collections import OrderedDict
import numpy as np
import torch
from scripts.skeleton_common import ROOT,CACHE,POSE,PROTOCOL,digest,read,sha,seal,offline,rgb_tensor
from scripts.skeleton_io import write
from scripts.skeleton_round2 import OUT,MANIFESTS,POLICY,require_source_seal,choose_threshold,selection_key
from scripts.train_skeleton_comparison import make_rgb,PRETRAIN,save_torch,training_mode
from backend.vision.skeleton_actions import SkeletonActionModel,prepare_skeleton


def rng_state():
    n=np.random.get_state()
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),python=random.getstate(),
        numpy=dict(name=n[0],keys=n[1].tolist(),position=n[2],has_gauss=n[3],cached_gauss=n[4]))


def restore_rng(value):
    torch.set_rng_state(value['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in value['cuda']])
    random.setstate(value['python']);n=value['numpy']
    np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gauss']))


def tensor_bytes(value):
    if torch.is_tensor(value):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(tensor_bytes(v) for v in value)
    return 0


class SampleStore:
    """At most 128 MiB/8 prepared samples in CPU LRU; no dataset-sized GPU state."""
    def __init__(self,name,rows,modality,allow_feature_build=False):
        self.name=name;self.rows=rows;self.modality=modality;self.pose_sha=sha(POSE)
        self.pretrain_sha=sha(PRETRAIN) if modality=='rgb' else None
        self.feature_signature=digest(dict(version=2,pretrain=self.pretrain_sha,protocol=PROTOCOL,
            transform='RGB uint8 letterbox112; first-round rgb_tensor',precision='CUDA autocast fp16',
            trunk='frozen Kinetics stem through layer3, layer4 pooled for head-only stage',
            transform_code=sha(ROOT/'scripts/skeleton_common.py'),rgb_factory_code=sha(ROOT/'scripts/train_skeleton_comparison.py')))
        self.folder=CACHE/'round2/rgb_features'/name/self.feature_signature[:16]
        self.lru=OrderedDict();self.bytes=0;self.max_bytes=128*1024*1024;self.max_items=8
        self.allow_feature_build=allow_feature_build;self.extractor=None
        self.expected={r['sample_id']:digest(dict(sample=r['sample_id'],source=r['sha256'],protocol=PROTOCOL,pose=self.pose_sha,extractor_version=1)) for r in rows}

    def verify_sources(self):
        """No test model access: this verifies only immutable file provenance."""
        start=time.monotonic()
        for i,r in enumerate(self.rows):
            if sha(r['path'])!=r['sha256']:raise ValueError('Source bytes changed: '+r['path'])
            if (i+1)%500==0 or i+1==len(self.rows):
                print('PROVENANCE',self.name,i+1,'/',len(self.rows),'seconds',round(time.monotonic()-start,2),flush=True)

    def raw(self,row):
        path=CACHE/self.name/(row['sample_id']+'.pt')
        if not path.is_file():return None
        saved=torch.load(path,map_location='cpu',weights_only=True)
        wanted=dict(signature=self.expected[row['sample_id']],source_sha256=row['sha256'],pose_sha256=self.pose_sha,protocol=PROTOCOL,status='complete')
        if any(saved.get(k)!=v for k,v in wanted.items()):raise ValueError('Cache provenance mismatch: '+str(path))
        return saved

    def feature_file(self,row):return self.folder/(row['sample_id']+'.pt')

    def make_features(self,row,cache):
        if not self.allow_feature_build:raise ValueError('RGB features must be prepared before action initialization/training')
        if self.extractor is None:self.extractor=make_rgb().cuda().eval()
        l3=[];pooled=[]
        with torch.inference_mode(),torch.autocast('cuda'):
            for clip in cache['clips']:
                model=self.extractor;x=rgb_tensor(clip['rgb']).unsqueeze(0).cuda()
                z=model.layer3(model.layer2(model.layer1(model.stem(x))))
                y=model.avgpool(model.layer4(z)).flatten(1)
                l3.append(z[0].cpu().half());pooled.append(y[0].cpu().float())
        saved=dict(signature=self.feature_signature,source_signature=cache['signature'],
            layer3=torch.stack(l3),pooled=torch.stack(pooled))
        save_torch(self.feature_file(row),saved)
        return saved

    def get(self,row):
        sid=row['sample_id']
        if sid in self.lru:
            self.lru.move_to_end(sid);return self.lru[sid][0]
        if self.modality=='rgb' and self.feature_file(row).exists():
            data=torch.load(self.feature_file(row),map_location='cpu',weights_only=True)
            if data['signature']!=self.feature_signature or data['source_signature']!=self.expected[sid]:raise ValueError('RGB feature signature mismatch')
        else:
            cache=self.raw(row)
            if cache is None:data=None
            elif self.modality=='rgb':data=self.make_features(row,cache)
            else:data=[prepare_skeleton(clip,'cpu') for clip in cache['clips']]
        size=tensor_bytes(data)
        if size<=self.max_bytes:
            while self.lru and (self.bytes+size>self.max_bytes or len(self.lru)>=self.max_items):
                _,(_,old)=self.lru.popitem(last=False);self.bytes-=old
            self.lru[sid]=(data,size);self.bytes+=size
        return data

    def close_extractor(self):
        self.extractor=None;self.lru.clear();self.bytes=0;torch.cuda.empty_cache()


def window_count(data,modality):return 0 if data is None else len(data['pooled']) if modality=='rgb' else len(data)


def window_logit(model,data,modality,stage,index):
    if modality=='skeleton':return model({k:v.cuda() for k,v in data[index].items()})
    if stage=='baseline':z=data['pooled'][index:index+1].cuda()
    else:z=model.avgpool(model.layer4(data['layer3'][index:index+1].float().cuda())).flatten(1)
    logits=model.fc(z);return logits[0,1]-logits[0,0]


def inference_logit(model,data,modality,stage):
    values=[]
    for i in range(window_count(data,modality)):
        value=window_logit(model,data,modality,stage,i)
        if value is not None:values.append(float(value))
    return max(values) if values else None


def training_logit(model,data,modality,stage):
    """Exact max-MIL with bounded activation memory, including tied maxima.

    Scan without gradients; replay winning windows with their original dropout
    RNG states. Frozen RGB batch-normalization and skeletal GroupNorm have no
    mutable running statistics. Restore post-scan RNG after the replays.
    """
    count=window_count(data,modality)
    if count==0:return None
    if count==1:return window_logit(model,data,modality,stage,0)
    winners=[];best=None
    with torch.no_grad():
        for i in range(count):
            before=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
            value=window_logit(model,data,modality,stage,i)
            if value is None:continue
            score=float(value)
            if not np.isfinite(score):raise ValueError('Nonfinite MIL score')
            if best is None or score>best:best=score;winners=[(i,before)]
            elif score==best:winners.append((i,before))
    after=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
    if not winners:return None
    # Usually exactly one winner; tied-window averaging reproduces torch.max's
    # subgradient rather than silently changing its treatment of score ties.
    values=[]
    try:
        for i,before in winners:
            torch.set_rng_state(before[0]);torch.cuda.set_rng_state_all(before[1])
            values.append(window_logit(model,data,modality,stage,i).float())
    finally:
        torch.set_rng_state(after[0]);torch.cuda.set_rng_state_all(after[1])
    return torch.stack(values).mean()


def evaluate(model,rows,store,modality,stage):
    model.eval();values=[]
    with torch.inference_mode(),torch.autocast('cuda'):
        for row in rows:
            logit=inference_logit(model,store.get(row),modality,stage)
            # Preserve first-round fp16-sigmoid score convention for both rounds.
            value=float(torch.tensor(logit,device='cuda',dtype=torch.float16).sigmoid()) if logit is not None else None
            values.append(value)
    return values


def configure(model,modality,stage):
    for p in model.parameters():p.requires_grad=modality=='skeleton'
    if modality=='rgb':
        for p in model.fc.parameters():p.requires_grad=True
        if stage=='finetune':
            for p in model.layer4.parameters():p.requires_grad=True
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001 if stage=='finetune' else .001,weight_decay=.0001)


def checkpoint(path,model,optimizer,scaler,state,signature):
    save_torch(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict() if optimizer else None,
        scaler=scaler.state_dict() if scaler else None,runner_state=copy.deepcopy(state),rng=rng_state(),config_signature=signature))


def run(args):
    offline();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no downloads permitted')
    manifest_path=Path(args.manifest) if args.manifest else MANIFESTS/f'{args.dataset}.json'
    manifest=require_source_seal(manifest_path)
    if manifest['dataset_key']!=args.dataset:raise ValueError('Dataset/manifest mismatch')
    out=Path(args.output_root) if args.output_root else OUT
    if out.resolve()==(ROOT/'results/video_events/skeleton_comparison').resolve():raise ValueError('First-round output directory prohibited')
    dest=out/args.dataset/args.modality/f'seed_{args.seed}';dest.mkdir(parents=True,exist_ok=True)
    with (dest/'train.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        train_locked(args,manifest,manifest_path,dest)


def train_locked(args,manifest,manifest_path,dest):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    store=SampleStore(args.dataset,manifest['rows'],args.modality,allow_feature_build=True)
    groups={s:[r for r in manifest['rows'] if r['split']==s] for s in ['train','validation']}
    config=dict(dataset=args.dataset,modality=args.modality,seed=args.seed,manifest_sha256=sha(manifest_path),
        source_seal_sha256=sha(manifest_path.with_suffix('.seal.json')),policy=POLICY,pose_sha256=sha(POSE),
        initialization='local Kinetics R3D-18' if args.modality=='rgb' else 'random skeleton action weights',
        pretrain_sha256=sha(PRETRAIN) if args.modality=='rgb' else None,sampling_protocol=PROTOCOL,
        optimizer='AdamW',learning_rates={'baseline':.001,'finetune':.0001},weight_decay=.0001,
        gradient_accumulation=4,gradient_clip=5.,class_weight='train normal count / train positive count, including unusable source rows',
        activation_memory='no-grad window scan + exact RNG replay of max windows',lru_bytes=store.max_bytes,lru_items=store.max_items,
        numerical_policy='CUDA autocast fp16; fp16 sigmoid; deterministic algorithms; TF32 disabled',
        resume='optimizer-step boundary at >=60 seconds and each epoch; optimizer/scaler/RNG/order position/metrics/signature',
        code_hashes={f:sha(ROOT/f) for f in ['scripts/train_skeleton_round2.py','scripts/skeleton_round2.py','scripts/train_skeleton_comparison.py','scripts/skeleton_common.py','backend/vision/skeleton_actions.py']})
    seal(dest/'config.json',config);signature=digest(config)
    if (dest/'selection.json').exists():
        selection=read(dest/'selection.json')
        if selection['config_signature']!=signature or sha(dest/'selected_best.pt')!=selection['sha256']:raise ValueError('Completed model provenance mismatch')
        print('ALREADY COMPLETE',args.dataset,args.modality,args.seed,flush=True);return
    store.verify_sources()
    if args.modality=='rgb':
        start=time.monotonic();rows=groups['train']+groups['validation']
        for i,r in enumerate(rows):
            store.get(r)
            if (i+1)%100==0 or i+1==len(rows):
                progress=dict(phase='prepare_frozen_rgb_features',completed=i+1,total=len(rows),dataset=args.dataset,modality=args.modality,seed=args.seed,epoch=0,seconds=time.monotonic()-start)
                write(dest/'progress.json',progress);print(progress,flush=True)
        store.close_extractor()
    store.allow_feature_build=False
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    model=(make_rgb() if args.modality=='rgb' else SkeletonActionModel(manifest['task'])).cuda()
    stages=list(POLICY['stage_epoch_limits'][args.modality]);limits=POLICY['stage_epoch_limits'][args.modality]
    state=dict(stage_index=0,epoch=0,cursor=0,epochs_total=0,patience=0,loss_sum=0.,used=0,skipped=[],histories={s:[] for s in stages},best_keys={},phase='train',epoch_seconds_accumulated=0.)
    resume=dest/'resume.pt';saved=None
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        if saved['config_signature']!=signature:raise ValueError('Resume config signature mismatch')
        state=saved['runner_state'];model.load_state_dict(saved['model'])
    positive=sum(r['label'] for r in groups['train']);negative=len(groups['train'])-positive
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative/positive,device='cuda'))
    while state['stage_index']<len(stages):
        stage=stages[state['stage_index']]
        optimizer=configure(model,args.modality,stage);scaler=torch.amp.GradScaler('cuda')
        if saved:
            if saved['optimizer'] is not None:optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
            restore_rng(saved['rng']);saved=None
        last_save=time.monotonic()
        while state['epoch']<limits[stage] and state['patience']<8:
            epoch_start=time.monotonic();training_mode(model,args.modality,stage);optimizer.zero_grad(set_to_none=True);pending=0
            order=torch.randperm(len(groups['train']),generator=torch.Generator().manual_seed(args.seed*1000+state['epoch'])).tolist()
            for position in range(state['cursor'],len(order)):
                row=groups['train'][order[position]]
                with torch.autocast('cuda'):
                    value=training_logit(model,store.get(row),args.modality,stage)
                    if value is not None:loss=criterion(value,torch.tensor(float(row['label']),device='cuda'))
                state['cursor']=position+1
                if value is None:state['skipped'].append(row['sample_id'])
                else:
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    scaler.scale(loss/4).backward();pending+=1;state['loss_sum']+=float(loss.detach());state['used']+=1
                if pending==4 or (position==len(order)-1 and pending):
                    scaler.unscale_(optimizer)
                    if pending!=4:
                        for p in model.parameters():
                            if p.grad is not None:p.grad.mul_(4/pending)
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);pending=0
                if (position+1)%100==0 or position==len(order)-1:
                    progress=dict(phase='train',dataset=args.dataset,modality=args.modality,seed=args.seed,stage=stage,
                        epoch=state['epoch']+1,cumulative_epoch=state['epochs_total']+1,max_total_epochs=50,completed=position+1,total=len(order),used=state['used'],unknown=len(state['skipped']),
                        elapsed_epoch_seconds=state['epoch_seconds_accumulated']+time.monotonic()-epoch_start)
                    write(dest/'progress.json',progress);print(progress,flush=True)
                if pending==0 and time.monotonic()-last_save>=60:
                    state['epoch_seconds_accumulated']+=time.monotonic()-epoch_start;epoch_start=time.monotonic()
                    checkpoint(resume,model,optimizer,scaler,state,signature);last_save=time.monotonic()
            if state['used']==0:raise RuntimeError('No usable training samples')
            state['phase']='validation';checkpoint(resume,model,optimizer,scaler,state,signature)
            vals=evaluate(model,groups['validation'],store,args.modality,stage)
            threshold,metric=choose_threshold([r['label'] for r in groups['validation']],vals)
            key=selection_key(metric)
            record=dict(stage=stage,epoch=state['epoch']+1,cumulative_epoch=state['epochs_total']+1,
                threshold=threshold,validation=metric,loss=state['loss_sum']/state['used'],used_training_segments=state['used'],unusable_training_segments=state['skipped'],
                seconds=state['epoch_seconds_accumulated']+time.monotonic()-epoch_start)
            state['histories'][stage].append(record)
            if stage not in state['best_keys'] or key>tuple(state['best_keys'][stage]):
                state['best_keys'][stage]=list(key);state['patience']=0
                best=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config,
                    config_signature=signature,stage=stage,epoch=state['epoch']+1,cumulative_epoch=state['epochs_total']+1,threshold=threshold,validation=metric,labels=manifest['labels'])
                save_torch(dest/f'{stage}_best.pt',best)
                write(dest/f'{stage}_validation_predictions.json',dict(threshold=threshold,rows=[dict(sample_id=r['sample_id'],label=r['label'],score=v) for r,v in zip(groups['validation'],vals)]))
            else:state['patience']+=1
            state.update(epoch=state['epoch']+1,epochs_total=state['epochs_total']+1,cursor=0,loss_sum=0.,used=0,skipped=[],phase='train',epoch_seconds_accumulated=0.)
            checkpoint(resume,model,optimizer,scaler,state,signature)
            write(dest/'run_record.json',dict(status='training',config_signature=signature,**state))
            print('EPOCH COMPLETE',args.dataset,args.modality,args.seed,record,flush=True)
            if state['epochs_total']>50:raise AssertionError('Cumulative epoch budget exceeded')
        state['stage_index']+=1;state.update(epoch=0,cursor=0,patience=0,phase='train')
        if state['stage_index']<len(stages):
            model.load_state_dict(torch.load(dest/'baseline_best.pt',map_location='cpu',weights_only=True)['state_dict'])
        checkpoint(resume,model,None,None,state,signature)
    selected=max(stages,key=lambda s:tuple(state['best_keys'][s]))
    best=torch.load(dest/f'{selected}_best.pt',map_location='cpu',weights_only=True)
    save_torch(dest/'selected_best.pt',best)
    write(dest/'selection.json',dict(status='complete',config_signature=signature,selected_stage=selected,epoch=best['epoch'],
        cumulative_epoch=best['cumulative_epoch'],epochs_trained=state['epochs_total'],threshold=best['threshold'],validation=best['validation'],
        criterion='validation event recall at normal FPR<=5%, then lower FPR, then macro F1; earlier epoch/stage on tie',
        sha256=sha(dest/'selected_best.pt'),checkpoint=str(dest/'selected_best.pt'),test_accessed=False))
    write(dest/'run_record.json',dict(status='complete',config_signature=signature,**state))
    write(dest/'progress.json',dict(phase='complete',dataset=args.dataset,modality=args.modality,seed=args.seed,epochs_trained=state['epochs_total'],test_accessed=False))
    print('MODEL COMPLETE',args.dataset,args.modality,args.seed,'epochs',state['epochs_total'],'validation',best['validation'],flush=True)


if __name__=='__main__':
    # Required by deterministic CUDA BLAS. Set before the first CUDA context.
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['fallvision','vfd'],required=True)
    p.add_argument('--modality',choices=['rgb','skeleton'],required=True);p.add_argument('--seed',type=int,choices=[42,43,44],required=True)
    p.add_argument('--manifest');p.add_argument('--output-root');run(p.parse_args())
