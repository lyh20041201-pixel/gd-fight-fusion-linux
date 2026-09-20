"""Train all six paired models, seal selections, then evaluate held-out data."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import time
from scripts.fine_actions_bc import *


def train(arm,seed):
    setup();plan=verify_seal();dest=OUT/arm/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'selection.json').exists():
        old=read(dest/'selection.json');assert sha(dest/'best.pt')==old['checkpoint_sha256'];return
    train_data=load_data('train');val_data=load_data('validation')
    usable=[w for w in train_data['windows'] if w['feature_index']>=0 and w['label']>=0]
    per_video=Counter(w['video_id'] for w in usable)
    rawweights=np.asarray([1/per_video[w['video_id']] for w in usable],np.float32)
    binary=np.asarray([int(w['label']==0) for w in usable])
    for value in [0,1]:rawweights[binary==value]/=rawweights[binary==value].sum()
    rawweights/=rawweights.mean()
    weights=torch.from_numpy(rawweights).cuda()
    target=torch.tensor([w['label'] if arm=='C' else int(w['label']!=0) for w in usable],device='cuda')
    idx=[w['feature_index'] for w in usable]
    x=train_data['x'][idx].cuda();valid=train_data['valid'][idx].cuda()
    model,initialization=make_model(arm,seed);seal(dest/'initialization.json',initialization);model=model.cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=POLICY['learning_rate'],weight_decay=POLICY['weight_decay'])
    scaler=torch.amp.GradScaler('cuda');history=[];best_key=None;stale=0;start_epoch=0
    if (dest/'resume.pt').exists():
        saved=torch.load(dest/'resume.pt',weights_only=True,map_location='cpu')
        assert saved['protocol_sha256']==sha(OUT/'protocol_frozen.json')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        history=saved['history'];best_key=tuple(saved['best_key']);stale=saved['stale'];start_epoch=saved['epoch']
    for epoch in range(start_epoch,POLICY['max_epochs']):
        seed_all(seed*1000+epoch);started=time.monotonic();model.train();loss_sum=0;count=0
        order=torch.randperm(len(usable),generator=torch.Generator().manual_seed(seed*1000+epoch)).tolist()
        for begin in range(0,len(order),POLICY['batch_size']):
            ids=order[begin:begin+POLICY['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(x[ids],valid[ids]);loss=(F.cross_entropy(logits,target[ids],reduction='none')*weights[ids]).mean()
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),POLICY['gradient_clip'])
            scaler.step(optimizer);scaler.update();loss_sum+=float(loss.detach())*len(ids);count+=len(ids)
        predictions=predict(model,val_data)
        threshold,metric,key=select_threshold(val_data['videos'],predictions)
        record=dict(epoch=epoch+1,loss=loss_sum/count,validation={k:v for k,v in metric.items() if k not in ['videos','per_action_normal_videos']},
            binary_validation_nll=binary_nll(predictions),threshold=threshold,seconds=time.monotonic()-started)
        history.append(record)
        if best_key is None or key>best_key:
            best_key=key;stale=0
            save_torch(dest/'best.pt',dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,
                epoch=epoch+1,threshold=threshold,protocol_sha256=sha(OUT/'protocol_frozen.json')))
            write(dest/'validation_predictions.json',dict(threshold=threshold,metrics=metric,predictions=predictions))
        else:stale+=1
        save_torch(dest/'resume.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            history=history,best_key=list(best_key),stale=stale,epoch=epoch+1,protocol_sha256=sha(OUT/'protocol_frozen.json')))
        write(dest/'history.json',history)
        progress=dict(arm=arm,seed=seed,epoch=epoch+1,max_epochs=POLICY['max_epochs'],validation_tp=metric['tp'],
            validation_fn=metric['fn'],validation_fp=metric['fp'],loss=record['loss'],seconds=record['seconds'],stale=stale)
        write(OUT/'progress.json',dict(status='training',**progress));print('EPOCH',progress,flush=True)
        if epoch+1>=POLICY['min_epochs'] and stale>=POLICY['patience']:break
    best=torch.load(dest/'best.pt',weights_only=True,map_location='cpu')
    result=dict(arm=arm,seed=seed,selected_epoch=best['epoch'],epochs_trained=len(history),threshold=best['threshold'],
                checkpoint_sha256=sha(dest/'best.pt'),protocol_sha256=sha(OUT/'protocol_frozen.json'),
                validation=read(dest/'validation_predictions.json')['metrics'],test_accessed_during_selection=False)
    seal(dest/'selection.json',result);print('COMPLETE',arm,seed,'selected',best['epoch'],flush=True)


def evaluate_all():
    setup();verify_seal()
    selections={f'{a}_{s}':read(OUT/a/f'seed_{s}'/'selection.json') for s in POLICY['seeds'] for a in POLICY['arms']}
    seal(OUT/'all_selections_before_test.json',selections)
    test=load_data('test');summary={}
    for arm in ['A','B','C']:
        for seed in ([None] if arm=='A' else POLICY['seeds']):
            if arm=='A':model,spec=make_baseline();threshold=spec['threshold'];name='A'
            else:
                name=f'{arm}_{seed}';saved=torch.load(OUT/arm/f'seed_{seed}'/'best.pt',weights_only=True,map_location='cpu')
                assert sha(OUT/arm/f'seed_{seed}'/'best.pt')==selections[name]['checkpoint_sha256']
                model=FineActionModel(2 if arm=='B' else len(CLASSES));model.load_state_dict(saved['state_dict']);threshold=saved['threshold']
            predictions=predict(model.cuda(),test);m=metrics(test['videos'],predictions,threshold)
            m['threshold']=threshold;m['binary_nll_vs_AI_window_labels']=binary_nll(predictions)
            if arm=='C':
                cm=np.zeros((8,8),int)
                for w in predictions:
                    if w['score'] is not None and w['label']>=0:cm[w['label'],int(np.argmax(w['probabilities']))]+=1
                m['AI_action_confusion_matrix']=cm.tolist()
            summary[name]=m
            write(OUT/'evaluation'/f'{name}_gmd_test.json',dict(metrics=m,predictions=predictions))
            print('TEST',name,'TP',m['tp'],'FN',m['fn'],'FP',m['fp'],'TN',m['tn'],'unknown',m['all_unknown_videos'],flush=True)
            del model
    write(OUT/'gmd_test_summary.json',summary)
    write(OUT/'progress.json',dict(status='gmd_evaluation_complete',next='external regression and report'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','train','evaluate','all']);p.add_argument('--arm',choices=['B','C']);p.add_argument('--seed',type=int)
    args=p.parse_args()
    if args.command=='freeze':freeze()
    elif args.command=='train':train(args.arm,args.seed)
    elif args.command=='evaluate':evaluate_all()
    else:
        for seed in POLICY['seeds']:
            for arm in POLICY['arms']:train(arm,seed)
        evaluate_all()
