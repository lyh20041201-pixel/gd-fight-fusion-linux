"""One-seed exploratory RGB head refinement, selected using validation only.

Reuse the existing RGB/42 backbone; emphasize difficult *training* negatives.
Old models, extraction caches and evaluations remain immutable.
"""
from pathlib import Path
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,json,time
import numpy as np
import torch
from scripts.skeleton_common import ROOT,CACHE,read,sha,offline
from scripts.skeleton_io import write
from scripts.skeleton_round2 import choose_threshold,selection_key,from_confusion
from scripts.train_skeleton_round2 import SampleStore
from scripts.train_skeleton_comparison import make_rgb,save_torch

OUT=ROOT/'results/live_actions/rgb_hard_negatives_v1'
BASE=ROOT/'results/video_events/skeleton_comparison_round2/vfd/rgb/seed_42'

def main():
    offline();torch.set_num_threads(4);torch.manual_seed(4201);np.random.seed(4201)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'selection.json').exists():raise RuntimeError('Experiment is already sealed; use a new version')
    base=torch.load(BASE/'selected_best.pt',map_location='cpu',weights_only=True)
    model=make_rgb(False);model.load_state_dict(base['state_dict']);model=model.cuda().eval()
    rows=read(CACHE/'round2/manifests/vfd.json')['rows'];store=SampleStore('vfd',rows,'rgb')
    signature=sha(BASE/'selected_best.pt')
    def features(row):
        dest=OUT/'features'/(row['sample_id']+'.pt')
        if dest.exists():
            value=torch.load(dest,map_location='cpu',weights_only=True)
            assert value['base_sha256']==signature
            return value['pooled']
        data=store.get(row);pooled=[]
        with torch.inference_mode(),torch.autocast('cuda'):
            for start in range(0,len(data['layer3']),4):
                # Use one clip per invocation to preserve batch-dependent kernels.
                for item in data['layer3'][start:start+4]:
                    pooled.append(model.avgpool(model.layer4(item[None].float().cuda())).flatten(1)[0].cpu().float())
        result=torch.stack(pooled)
        save_torch(dest,dict(base_sha256=signature,pooled=result))
        return result
    data={}
    development=[r for r in rows if r['split'] in ('train','validation')]
    for i,row in enumerate(development):
        data[row['sample_id']]=features(row)
        if (i+1)%100==0:print('FEATURES',i+1,len(development),flush=True)
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation']
    def scores(population):
        result=[]
        with torch.inference_mode(),torch.autocast('cuda'):
            for row in population:
                # Match the original batch-one validation path exactly.
                values=[]
                for pooled in data[row['sample_id']]:
                    logits=model.fc(pooled[None].cuda())
                    values.append(float((logits[:,1]-logits[:,0]).sigmoid()[0]))
                result.append(max(values))
        return result
    initial=scores(val)
    # Use the independently recomputed deployment baseline (all 207 samples).
    # Historical scores can differ by one fp16 rounding step on this runtime.
    expected=[dict(sample_id=r['sample_id'],score=max(r['windows'][0]))
              for r in read(ROOT/'results/live_actions/rgb_ensemble_v1/validation_windows.json')['rows']]
    assert [r['sample_id'] for r in val]==[r['sample_id'] for r in expected]
    if initial!=[r['score'] for r in expected]:
        differences=[dict(sample_id=r['sample_id'],expected=r['score'],actual=v) for r,v in zip(expected,initial) if v!=r['score']]
        write(OUT/'feature_parity_differences.json',dict(rows=differences))
        raise AssertionError(f'Feature extraction parity failed: {len(differences)} differences; first={differences[0]}')
    train_scores=scores(train)
    normal=[(score,r['sample_id']) for r,score in zip(train,train_scores) if r['label']==0]
    hard={sid for _,sid in sorted(normal,reverse=True)[:max(1,len(normal)//10)]}
    protocol=dict(base_sha256=signature,seed=4201,max_epochs=20,patience=5,
        optimizer='AdamW',learning_rate=.0001,weight_decay=.0001,
        trained_parameters='fc only; all backbone layers frozen',
        hard_negatives='top 10% training normal segments by initial score; weight 3',
        hard_negative_ids=sorted(hard),validation_only_selection=True,
        test_has_prior_evaluation_history=True,code_sha256=sha(__file__))
    write(OUT/'protocol.json',protocol)
    threshold,metric=choose_threshold([r['label'] for r in val],initial)
    best_key=selection_key(metric);best=copy.deepcopy(base);best.update(threshold=threshold,validation=metric)
    best_epoch=0;best_scores=initial;patience=0;history=[]
    optimizer=torch.optim.AdamW(model.fc.parameters(),lr=.0001,weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda')
    for epoch in range(1,21):
        total=0.;optimizer.zero_grad();pending=0
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(4201000+epoch)).tolist()
        for position,index in enumerate(order):
            row=train[index]
            with torch.autocast('cuda'):
                logits=model.fc(data[row['sample_id']].cuda())
                value=(logits[:,1]-logits[:,0]).max()
                weight=3. if row['sample_id'] in hard else 1.
                loss=torch.nn.functional.binary_cross_entropy_with_logits(value,torch.tensor(float(row['label']),device='cuda'))*weight
            scaler.scale(loss/16).backward();total+=float(loss.detach());pending+=1
            if pending==16 or position==len(order)-1:
                scaler.unscale_(optimizer)
                if pending!=16:
                    for p in model.fc.parameters():p.grad.mul_(16/pending)
                torch.nn.utils.clip_grad_norm_(model.fc.parameters(),5)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad();pending=0
        values=scores(val);threshold,metric=choose_threshold([r['label'] for r in val],values)
        record=dict(epoch=epoch,loss=total/len(train),threshold=threshold,validation=metric)
        history.append(record);write(OUT/'history.json',dict(history=history))
        print('EPOCH',epoch,'recall',metric['recall'][1],'FPR',metric['normal_false_positive_rate'],flush=True)
        key=selection_key(metric)
        if key>best_key:
            best_key=key;best_epoch=epoch;best_scores=values;patience=0
            best=copy.deepcopy(base);best.update(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},threshold=threshold,validation=metric)
        else:patience+=1
        if patience>=5:break
    save_torch(OUT/'selected_best.pt',best)
    selection=dict(status='sealed',improved_on_validation=best_epoch>0,epoch=best_epoch,
        threshold=best['threshold'],validation=best['validation'],baseline_validation=base['validation'],
        checkpoint=str(OUT/'selected_best.pt'),sha256=sha(OUT/'selected_best.pt'),
        test_accessed_for_selection=False,selection_split='validation')
    write(OUT/'selection.json',selection)
    write(OUT/'validation_predictions.json',dict(rows=[dict(sample_id=r['sample_id'],label=r['label'],score=s) for r,s in zip(val,best_scores)]))
    print('SELECTED',json.dumps(selection),flush=True)
    model.load_state_dict(best['state_dict'])
    test=[r for r in rows if r['split']=='test']
    for row in test:data[row['sample_id']]=features(row)
    values=scores(test);cm=np.zeros((2,3),dtype=int)
    for row,value in zip(test,values):cm[row['label'],int(value>=best['threshold'])]+=1
    report=dict(metrics=from_confusion(cm),threshold=best['threshold'],threshold_retuned=False,
        prior_test_history=True,exploratory_single_seed=True,
        rows=[dict(sample_id=r['sample_id'],label=r['label'],score=v) for r,v in zip(test,values)])
    write(OUT/'heldout_evaluation.json',report)
    print('HELDOUT',json.dumps(report['metrics']),flush=True)

if __name__=='__main__':main()
