"""Three reproducible seeds, whole-segment MIL, validation-only model/threshold selection."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,time,copy
import torch,numpy as np
from torchvision.models.video import r3d_18
from backend.vision.skeleton_actions import SkeletonActionModel,prepare_skeleton
from scripts.skeleton_common import *

PRETRAIN=Path.home()/'.cache/torch/hub/checkpoints/r3d_18-b3b3357e.pth'

def save_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);temp.replace(path)

def make_rgb(pretrained=True):
    model=r3d_18(weights=None)
    if pretrained:
        if not PRETRAIN.is_file():raise ValueError('Local Kinetics checkpoint missing; automatic download prohibited')
        model.load_state_dict(torch.load(PRETRAIN,map_location='cpu',weights_only=True))
    model.fc=torch.nn.Linear(512,2)
    return model

def load_rows(name):
    manifest=read(CACHE/'manifests'/f'{name}.json');rows=manifest['rows'];isolation(rows)
    caches={}
    for row in rows:
        path=CACHE/name/(row['sample_id']+'.pt')
        if path.exists():caches[row['sample_id']]=torch.load(path,map_location='cpu',weights_only=True)
        else:caches[row['sample_id']]=None
    return manifest,rows,caches

def rgb_features(name,rows,caches):
    signature=digest(dict(pretrained=sha(PRETRAIN),protocol=PROTOCOL,version=1))
    folder=CACHE/name/'rgb_features';folder.mkdir(parents=True,exist_ok=True)
    model=make_rgb().cuda().eval();output={}
    for i,row in enumerate(rows):
        sid=row['sample_id'];cache=caches[sid];path=folder/(sid+'.pt')
        if cache is None:output[sid]=None;continue
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved['signature']!=signature or saved['source_signature']!=cache['signature']:raise ValueError('RGB feature cache mismatch')
        else:
            l3=[];pooled=[]
            for clip in cache['clips']:
                with torch.inference_mode(),torch.autocast('cuda'):
                    x=rgb_tensor(clip['rgb']).unsqueeze(0).cuda()
                    z=model.layer3(model.layer2(model.layer1(model.stem(x))))
                    y=model.avgpool(model.layer4(z)).flatten(1)
                l3.append(z[0].cpu().half());pooled.append(y[0].cpu().float())
            saved=dict(signature=signature,source_signature=cache['signature'],layer3=torch.stack(l3),pooled=torch.stack(pooled))
            save_torch(path,saved)
        output[sid]=saved
        if i%25==0:print('RGB feature cache',name,i+1,'/',len(rows),flush=True)
    del model;torch.cuda.empty_cache();return output

def score(model,data,modality,stage):
    if data is None:return None
    if modality=='rgb':
        if stage=='baseline':features=data['pooled'].to('cuda')
        else:features=model.avgpool(model.layer4(data['layer3'].float().cuda())).flatten(1)
        logits=model.fc(features);return (logits[:,1]-logits[:,0]).max()
    logits=[model(clip) for clip in data]
    logits=[v for v in logits if v is not None]
    return torch.stack(logits).max() if logits else None

def evaluate(model,rows,data,modality,stage):
    model.eval();scores=[]
    with torch.inference_mode():
        for row in rows:
            with torch.autocast('cuda'):
                value=score(model,data[row['sample_id']],modality,stage)
            scores.append(float(value.sigmoid()) if value is not None else None)
    return scores

def training_mode(model,modality,stage):
    if modality=='skeleton':model.train();return
    model.eval();model.fc.train()
    if stage=='finetune':
        model.layer4.train()
        for module in model.layer4.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['gmd','tnue'],required=True)
    p.add_argument('--modality',choices=['rgb','skeleton'],required=True);p.add_argument('--seed',type=int,choices=[42,43,44],required=True)
    a=p.parse_args();offline();torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    task='fall' if a.dataset=='gmd' else 'fight';out=OUT/a.dataset/a.modality/f'seed_{a.seed}'
    out.mkdir(parents=True,exist_ok=True)
    manifest,rows,caches=load_rows(a.dataset)
    config=dict(dataset=a.dataset,task=task,modality=a.modality,seed=a.seed,protocol=PROTOCOL,
                manifest_sha256=sha(CACHE/'manifests'/f'{a.dataset}.json'),pose_sha256=sha(POSE),
                architecture='r3d_18' if a.modality=='rgb' else 'stgcn_6blocks_32_32_64_64_128_128_symmetric_pair_head',
                initialization='local Kinetics-400 R3D-18' if a.modality=='rgb' else 'random action weights; frozen pretrained pose extractor',
                optimizer='AdamW',weight_decay=.0001,gradient_accumulation_segments=4,
                annotation_status=manifest['label_status'],label_limitation=manifest['limitation'],
                learning_rates={'baseline':.001,'finetune':.0001},
                code_hashes={f:sha(ROOT/f) for f in ['scripts/train_skeleton_comparison.py','backend/vision/skeleton_actions.py','scripts/skeleton_common.py']})
    seal(out/'config.json',config)
    if (out/'selection.json').exists():print('already complete',out,flush=True);return
    if a.modality=='rgb':data=rgb_features(a.dataset,rows,caches)
    else:data={sid:([prepare_skeleton(c,'cuda') for c in cache['clips']] if cache else None) for sid,cache in caches.items()}
    del caches
    # Feature construction must not perturb each seed's action-model initialization.
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    model=(make_rgb() if a.modality=='rgb' else SkeletonActionModel(task)).cuda()
    groups={s:[r for r in rows if r['split']==s] for s in ('train','validation','test')}
    stages=['baseline','finetune'] if a.modality=='rgb' else ['baseline']
    state=read(out/'run_record.json') if (out/'run_record.json').exists() else dict(status='running',config=config,stages={})
    positive=sum(r['label'] for r in groups['train']);negative=len(groups['train'])-positive
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative/max(1,positive),device='cuda'))
    for stage in stages:
        if state['stages'].get(stage,{}).get('status')=='complete':continue
        for p0 in model.parameters():p0.requires_grad=a.modality=='skeleton'
        if a.modality=='rgb':
            for p0 in model.fc.parameters():p0.requires_grad=True
            if stage=='finetune':
                model.load_state_dict(torch.load(out/'baseline_best.pt',weights_only=True)['state_dict'])
                for p0 in model.layer4.parameters():p0.requires_grad=True
        optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=config['learning_rates'][stage],weight_decay=.0001)
        scaler=torch.amp.GradScaler('cuda');resume=out/f'{stage}_resume.pt'
        history=[];best=-1.;patience=0;start_epoch=0
        if resume.exists():
            saved=torch.load(resume,weights_only=True)
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
            history=saved['history'];best=saved['best'];patience=saved['patience'];start_epoch=saved['next_epoch']
            torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        for epoch in range(start_epoch,50):
            if patience>=8:break
            started=time.monotonic();training_mode(model,a.modality,stage);optimizer.zero_grad(set_to_none=True)
            order=torch.randperm(len(groups['train']),generator=torch.Generator().manual_seed(a.seed*1000+epoch)).tolist()
            losses=[];skipped=[];pending=0
            for idx in order:
                row=groups['train'][idx]
                with torch.autocast('cuda'):
                    value=score(model,data[row['sample_id']],a.modality,stage)
                    if value is None:skipped.append(row['sample_id']);continue
                    loss=criterion(value,torch.tensor(float(row['label']),device='cuda'))
                if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                scaler.scale(loss/4).backward();pending+=1;losses.append(float(loss.detach()))
                if pending==4:
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],5.)
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);pending=0
            if pending:
                scaler.unscale_(optimizer)
                for p0 in model.parameters():
                    if p0.grad is not None:p0.grad.mul_(4/pending)
                torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],5.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            if not losses:raise RuntimeError('No usable training samples')
            vals=evaluate(model,groups['validation'],data,a.modality,stage)
            threshold,validation=choose_threshold([r['label'] for r in groups['validation']],vals)
            result=dict(epoch=epoch+1,loss=float(np.mean(losses)),validation=validation,threshold=threshold,
                        used_training_segments=len(losses),unusable_training_segments=skipped,seconds=time.monotonic()-started)
            history.append(result)
            if validation['macro_f1']>best+1e-8:
                best=validation['macro_f1'];patience=0
                checkpoint=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config,
                                threshold=threshold,epoch=epoch+1,validation=validation,stage=stage,labels=manifest['labels'])
                save_torch(out/f'{stage}_best.pt',checkpoint)
            else:patience+=1
            save_torch(resume,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                                   history=history,best=best,patience=patience,next_epoch=epoch+1,
                                   rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
            state['stages'][stage]=dict(status='running',history=history,best_validation_macro_f1=best)
            write(out/'run_record.json',state)
            print(a.dataset,a.modality,a.seed,stage,'epoch',epoch+1,'val F1',round(validation['macro_f1'],4),'coverage',round(validation['coverage'],3),'loss',round(result['loss'],4),flush=True)
        state['stages'][stage]=dict(status='complete',history=history,best_validation_macro_f1=best)
        write(out/'run_record.json',state)
    selected=max(stages,key=lambda s:state['stages'][s]['best_validation_macro_f1'])
    saved=torch.load(out/f'{selected}_best.pt',weights_only=True);model.load_state_dict(saved['state_dict'])
    values=evaluate(model,groups['test'],data,a.modality,selected);threshold=saved['threshold']
    predictions=[dict(sample_id=r['sample_id'],path=r['path'],label=r['label'],score=v,prediction=-1 if v is None else int(v>=threshold)) for r,v in zip(groups['test'],values)]
    result=metrics([r['label'] for r in groups['test']],[r['prediction'] for r in predictions])
    save_torch(out/'selected_best.pt',saved)
    write(out/'test_predictions.json',dict(predictions=predictions,metrics=result,threshold=threshold))
    state['status']='complete';state['selected_stage']=selected;state['test']=result;write(out/'run_record.json',state)
    write(out/'selection.json',dict(selected_stage=selected,criterion='validation macro F1 only',threshold=threshold,
                                  checkpoint=str(out/'selected_best.pt'),sha256=sha(out/'selected_best.pt'),test=result))
    print('COMPLETE',a.dataset,a.modality,a.seed,result,flush=True)

if __name__=='__main__':main()
