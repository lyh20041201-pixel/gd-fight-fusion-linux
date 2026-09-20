"""R3D-18 baseline and last-stage fine tuning with sealed test split."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse, hashlib, json, random, time
import cv2
import numpy as np
import torch
from torchvision.models.video import r3d_18, R3D_18_Weights
from backend.vision.video_events import clip_tensor

def metrics(y,p,k):
    cm=np.zeros((k,k),int)
    for a,b in zip(y,p):cm[a,b]+=1
    pr=np.diag(cm)/np.maximum(1,cm.sum(0));re=np.diag(cm)/np.maximum(1,cm.sum(1))
    f=2*pr*re/np.maximum(1e-12,pr+re)
    return dict(precision=pr.tolist(),recall=re.tolist(),f1=f.tolist(),macro_f1=float(f.mean()),confusion_matrix=cm.tolist())

def decode(row):
    cap=cv2.VideoCapture(row['path'])
    fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps<=0 or n<=0:raise ValueError('Unreadable video: '+row['path'])
    start=float(row.get('start',0));end=min(float(row.get('end',n/fps)),n/fps)
    # One deterministic centred 4-second window per labelled clip/segment.
    start=max(start,(start+end)/2-2);end=min(end,start+4)
    ids=np.linspace(round(start*fps),max(round(start*fps),min(n-1,round(end*fps)-1)),16).astype(int)
    images=[]
    for idx in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx));ok,im=cap.read()
        if not ok:raise ValueError(f'Decode failed {row["path"]} frame {idx}')
        images.append(im)
    cap.release()
    return clip_tensor(images),dict(start=start,end=end,valid_duration=end-start)

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=50);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('GPU required for this experiment')
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh output directory; archived experiments cannot be overwritten')
    out.mkdir(parents=True,exist_ok=True)
    data=json.loads(Path(a.manifest).read_text(encoding='utf-8'));rows=data['rows'];labels=data['labels'];k=len(labels)
    if data.get('dataset') not in ('GMDCSA-24','TNUE-Fight Detection'):
        raise ValueError('Only the newly approved training datasets may be trained')
    for split in ('train','validation','test'):
        if set(r['label'] for r in rows if r['split']==split)!=set(range(k)):
            raise ValueError(f'Every split must contain all classes: {split}')
    groups={s:{r['group'] for r in rows if r['split']==s} for s in ['train','validation','test']}
    assert not(groups['train']&groups['test'] or groups['validation']&groups['test'] or groups['train']&groups['validation'])
    hashes={s:{r['sha256'] for r in rows if r['split']==s} for s in groups}
    assert not(hashes['train']&hashes['test'] or hashes['validation']&hashes['test'] or hashes['train']&hashes['validation'])
    record=dict(status='running',dataset=data['dataset'],protocol=data.get('protocol'),annotation_status=data.get('status','source_annotations_with_audited_corrections'),label_limitation=data.get('limitation'),labels=labels,manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
        device=torch.cuda.get_device_name(0),torch=str(torch.__version__),seed=42,window_seconds=4,frames=16,
        preprocessing='112 square letterbox RGB Kinetics normalization',split_counts={s:sum(r['split']==s for r in rows) for s in groups},
        batch_size=4,max_epochs=a.epochs,patience=8,stages={})
    def save(): (out/'run_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    save();cache=out/'clips.pt'
    (out/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    if cache.exists():x=torch.load(cache,weights_only=True)
    else:
        clips=[];windows=[]
        for i,row in enumerate(rows):
            clip,window=decode(row);clips.append(clip.half());windows.append(window)
            if i%25==0:print('decode',i,len(rows),flush=True)
        x=torch.stack(clips);torch.save(x,cache)
        (out/'windows.json').write_text(json.dumps(windows),encoding='utf-8')
    y=torch.tensor([r['label'] for r in rows]);indices={s:torch.tensor([i for i,r in enumerate(rows) if r['split']==s]) for s in groups}
    model=r3d_18(weights=R3D_18_Weights.KINETICS400_V1);model.fc=torch.nn.Linear(model.fc.in_features,k);model.cuda()
    for param in model.parameters():param.requires_grad=False
    for param in model.fc.parameters():param.requires_grad=True
    criterion=torch.nn.CrossEntropyLoss()
    def evaluate(ids):
        model.eval();pred=[];truth=[];prob=[];start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda'):
            for batch in ids.split(4):
                scores=model(x[batch].float().cuda()).softmax(1).cpu()
                pred+=scores.argmax(1).tolist();prob+=scores.tolist();truth+=y[batch].tolist()
        torch.cuda.synchronize()
        return metrics(truth,pred,k),prob,(time.perf_counter()-start)*1000/max(1,len(ids))
    for stage in ['baseline','finetune']:
        if stage=='finetune':
            saved=torch.load(out/'baseline_best.pt',weights_only=True);model.load_state_dict(saved['state_dict'])
            for param in model.layer4.parameters():param.requires_grad=True
        optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=.001 if stage=='baseline' else .0001,weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda');best=-1;wait=0;history=[]
        for epoch in range(a.epochs):
            model.eval();model.fc.train()
            if stage=='finetune':
                model.layer4.train()
                # Tiny batches must not corrupt pretrained running BN statistics.
                for module in model.layer4.modules():
                    if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
            order=indices['train'][torch.randperm(len(indices['train']))];losses=[]
            for batch in order.split(4):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda'):
                    loss=criterion(model(x[batch].float().cuda()),y[batch].cuda())
                scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();losses.append(float(loss))
            val,_,_=evaluate(indices['validation']);history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**val))
            print(stage,epoch+1,'valF1',val['macro_f1'],flush=True)
            if val['macro_f1']>best+1e-6:
                best=val['macro_f1'];wait=0
                torch.save(dict(state_dict={n:v.cpu() for n,v in model.state_dict().items()},labels=labels,threshold=.5,
                    dataset=data['dataset'],manifest_sha256=record['manifest_sha256'],architecture='r3d_18',
                    annotation_status=record['annotation_status'],label_limitation=record['label_limitation'],
                    pretrained='KINETICS400_V1',frames=16,window_seconds=4),out/f'{stage}_best.pt')
            else:wait+=1
            record['stages'][stage]=dict(history=history,best_validation_macro_f1=best);save()
            if wait>=8:break
        model.load_state_dict(torch.load(out/f'{stage}_best.pt',weights_only=True)['state_dict'])
        # Threshold selected strictly on validation, favour lower threshold on ties.
        _,probs,_=evaluate(indices['validation']);positive=np.array(probs)[:,1:].sum(1)
        target=(y[indices['validation']].numpy()>0).astype(int)
        threshold=max(np.arange(.1,.91,.05),key=lambda t:metrics(target,(positive>=t).astype(int),2)['macro_f1'])
        checkpoint=torch.load(out/f'{stage}_best.pt',weights_only=True);checkpoint['threshold']=float(threshold)
        torch.save(checkpoint,out/f'{stage}_best.pt')
        test,probs,ms=evaluate(indices['test'])
        threshold_test=metrics((y[indices['test']].numpy()>0).astype(int),(np.array(probs)[:,1:].sum(1)>=threshold).astype(int),2)
        record['stages'][stage].update(test=test,threshold_test=threshold_test,test_ms_per_clip=ms,threshold=float(threshold))
        predictions=[dict(path=rows[int(i)]['path'],start=rows[int(i)].get('start'),end=rows[int(i)].get('end'),group=rows[int(i)]['group'],label=int(y[i]),probabilities=pr,
            candidate=bool(sum(pr[1:])>=threshold)) for i,pr in zip(indices['test'],probs)]
        (out/f'{stage}_test_predictions.json').write_text(json.dumps(predictions,indent=2),encoding='utf-8');save()
    record['status']='complete';save()

if __name__=='__main__':main()
