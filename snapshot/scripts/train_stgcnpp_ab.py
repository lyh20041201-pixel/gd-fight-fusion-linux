"""Train one sealed, resumable ST-GCN++ A/B model without accessing heldout scores."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,msvcrt,time
from scripts.stgcnpp_ab import *

def run(dataset,arm,seed):
    setup_runtime()
    require_source_seal(MANIFESTS/f'{dataset}.json')
    prep=read(OUT/'audit/preparation_complete.json')
    frozen=read(OUT/'experiment_plan_frozen.json')
    if prep['feature_signature']!=feature_signature() or frozen['code_hashes']!={f:sha(ROOT/f) for f in CODE}:raise ValueError('Preparation/frozen plan mismatch')
    verify=read(OUT/'verification/preflight.json')
    if verify['status']!='passed' or verify['code_hashes']!={f:sha(ROOT/f) for f in CODE}:raise ValueError('Preflight missing or code changed')
    dest=OUT/dataset/arm/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
    with (dest/'train.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        train_locked(dataset,arm,seed,dest)

def train_locked(dataset,arm,seed,dest):
    manifest=require_source_seal(MANIFESTS/f'{dataset}.json');config=model_config(dataset,arm,seed);signature=digest(config)
    seal(dest/'config.json',config)
    if (dest/'selection.json').exists():
        old=read(dest/'selection.json')
        if old['config_signature']!=signature or sha(dest/'selected_best.pt')!=old['sha256']:raise ValueError('Completed artifact changed')
        print('ALREADY_COMPLETE',dataset,arm,seed,flush=True);return
    train=[r for r in manifest['rows'] if r['split']=='train'];val=[r for r in manifest['rows'] if r['split']=='validation']
    store=ABStore(dataset)
    model,init=make_model(manifest['task'],arm,seed);seal(dest/'initialization.json',init);model=model.cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=POLICY['learning_rate'],weight_decay=POLICY['weight_decay'])
    scaler=torch.amp.GradScaler('cuda')
    positive=sum(r['label'] for r in train)
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train)-positive)/positive,device='cuda'))
    state=dict(epoch=0,cursor=0,patience=0,loss_sum=0.,used=0,skipped=[],history=[],best_key=None,phase='train',epoch_seconds=0.,optimizer_steps=0)
    resume=dest/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        if saved['config_signature']!=signature:raise ValueError('Resume signature mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        state=saved['runner_state'];restore_rng(saved['rng'])
    last_save=time.monotonic()
    while state['epoch']<POLICY['max_total_epochs'] and state['patience']<POLICY['patience']:
        started=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True);pending=0
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed*1000+state['epoch'])).tolist()
        for position in range(state['cursor'],len(order)):
            row=train[order[position]];data=store.get(row)
            with torch.autocast('cuda'):
                value=training_logit(model,data)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);pending=0;state['optimizer_steps']+=1
            if (position+1)%50==0 or position==len(order)-1:
                progress=dict(phase='train',dataset=dataset,arm=arm,seed=seed,epoch=state['epoch']+1,max_epochs=50,completed=position+1,total=len(train),
                    used=state['used'],unknown=len(state['skipped']),seconds=state['epoch_seconds']+time.monotonic()-started)
                write(dest/'progress.json',progress);print(progress,flush=True)
            if pending==0 and time.monotonic()-last_save>=60:
                state['epoch_seconds']+=time.monotonic()-started;started=time.monotonic()
                checkpoint(resume,model,optimizer,scaler,state,signature);last_save=time.monotonic()
        if not state['used']:raise RuntimeError('No usable training samples')
        state['phase']='validation';checkpoint(resume,model,optimizer,scaler,state,signature)
        scores=evaluate(model,val,store,dict(path=dest/'progress.json',info=dict(dataset=dataset,arm=arm,seed=seed,epoch=state['epoch']+1,max_epochs=50)))
        threshold,metric=choose_threshold([r['label'] for r in val],scores);key=selection_key(metric)
        record=dict(epoch=state['epoch']+1,threshold=threshold,validation=metric,loss=state['loss_sum']/state['used'],used_training_segments=state['used'],
            unusable_training_segments=state['skipped'],seconds=state['epoch_seconds']+time.monotonic()-started,optimizer_steps=state['optimizer_steps'])
        state['history'].append(record)
        if state['best_key'] is None or key>tuple(state['best_key']):
            state['best_key']=list(key);state['patience']=0
            save_torch(dest/'best.pt',dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config,config_signature=signature,
                epoch=state['epoch']+1,threshold=threshold,validation=metric,labels=manifest['labels']))
            write(dest/'best_validation_predictions.json',dict(threshold=threshold,rows=[dict(sample_id=r['sample_id'],label=r['label'],score=s) for r,s in zip(val,scores)]))
        else:state['patience']+=1
        state.update(epoch=state['epoch']+1,cursor=0,loss_sum=0.,used=0,skipped=[],phase='train',epoch_seconds=0.)
        checkpoint(resume,model,optimizer,scaler,state,signature)
        write(dest/'run_record.json',dict(status='training',config_signature=signature,**state))
        print('EPOCH_COMPLETE',dataset,arm,seed,record['epoch'],'seconds',round(record['seconds'],2),'validation',metric,flush=True)
    best=torch.load(dest/'best.pt',map_location='cpu',weights_only=True);save_torch(dest/'selected_best.pt',best)
    write(dest/'selection.json',dict(status='complete',config_signature=signature,epoch=best['epoch'],epochs_trained=state['epoch'],threshold=best['threshold'],
        validation=best['validation'],sha256=sha(dest/'selected_best.pt'),checkpoint=str(dest/'selected_best.pt'),test_accessed=False))
    write(dest/'run_record.json',dict(status='complete',config_signature=signature,**state))
    write(dest/'progress.json',dict(phase='complete',dataset=dataset,arm=arm,seed=seed,epochs_trained=state['epoch'],selected_epoch=best['epoch']))
    print('MODEL_COMPLETE',dataset,arm,seed,'epochs',state['epoch'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=POLICY['datasets'],required=True);p.add_argument('--arm',choices=['A','B'],required=True)
    p.add_argument('--seed',type=int,choices=[42,43,44],required=True);a=p.parse_args();run(a.dataset,a.arm,a.seed)
