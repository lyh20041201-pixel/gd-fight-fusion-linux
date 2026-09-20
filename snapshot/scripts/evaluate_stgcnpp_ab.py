"""Compare frozen A/B selections on the retained test and original regression splits."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from scripts.stgcnpp_ab import *

KEYS=[f'{arm}_{seed}' for arm in POLICY['arms'] for seed in POLICY['seeds']]
ATTRS=['precision','recall','macro_f1','normal_false_positive_rate','coverage','confusion_matrix']

def freeze_selections():
    selected={}
    for name in POLICY['datasets']:
        for seed in POLICY['seeds']:
            initial=[]
            for arm in POLICY['arms']:
                folder=OUT/name/arm/f'seed_{seed}';config=model_config(name,arm,seed)
                s=read(folder/'selection.json');history=read(folder/'run_record.json')['history']
                assert s['status']=='complete' and s['config_signature']==digest(config) and s['sha256']==sha(folder/'selected_best.pt')
                assert len(history)==s['epochs_trained']<=50
                best=None;patience=0
                for h in history:
                    key=selection_key(h['validation'])
                    if best is None or key>selection_key(best['validation']):best=h;patience=0
                    else:patience+=1
                    if h is not history[-1]:assert patience<8
                assert best['epoch']==s['epoch'] and best['threshold']==s['threshold'] and (patience==8 or len(history)==50)
                pred=read(folder/'best_validation_predictions.json');threshold,metric=choose_threshold([r['label'] for r in pred['rows']],[r['score'] for r in pred['rows']])
                assert threshold==s['threshold'] and metric==s['validation']
                selected[f'{name}/{arm}/{seed}']=s;initial.append(read(folder/'initialization.json'))
            assert initial[0]['head_initial_sha256']==initial[1]['head_initial_sha256']
    value=dict(models=selected,model_count=12,selection_policy=POLICY['selection'],test_access_started=False,
        evaluation_code_sha256=sha(__file__),feature_signature=feature_signature())
    seal(OUT/'selection_seal.json',value)
    return value

def summarize(rows):
    per={k:metrics_for(rows,k) for k in KEYS};aggregate={};deltas={}
    for arm in POLICY['arms']:
        values=[per[f'{arm}_{seed}'] for seed in POLICY['seeds']]
        aggregate[arm]={a:dict(mean=np.asarray([m[a] for m in values],float).mean(0).tolist(),std=np.asarray([m[a] for m in values],float).std(0,ddof=1).tolist()) for a in ATTRS}
    for attr in ATTRS:
        values=np.asarray([np.asarray(per[f'B_{s}'][attr])-np.asarray(per[f'A_{s}'][attr]) for s in POLICY['seeds']])
        deltas[attr]=dict(per_seed=values.tolist(),mean=values.mean(0).tolist(),std=values.std(0,ddof=1).tolist())
    strata={}
    for attr in ['origin','person_size','pose_visibility_proxy','crowd']:
        def value(r):return r['origin'] if attr=='origin' else r['diagnostics'][attr]
        strata[attr]={v:dict(samples=sum(value(r)==v for r in rows),per_seed={k:metrics_for([r for r in rows if value(r)==v],k) for k in KEYS}) for v in sorted({value(r) for r in rows})}
    assert all(m['samples']==len(rows) for m in per.values())
    assert all(len({r['results'][k]['prediction']==-1 for k in KEYS})==1 for r in rows)
    return dict(samples=len(rows),sample_ids_digest=digest([r['sample_id'] for r in rows]),per_seed=per,aggregate=aggregate,paired_B_minus_A=deltas,strata=strata,
        same_denominators=True,same_unknown_samples=True,test_fpr_target_met={k:m['normal_false_positive_rate']<=.05 for k,m in per.items()},
        interpretation='Three-seed standard deviation uses ddof=1. Unknown positives count as missed events. Existing tests have prior experiment history, not new blind tests. Validation constraint does not guarantee test FPR.')

def evaluate_population(name,rows,models,thresholds,signature,dest,description):
    store=ABStore(name);records=[];started=time.monotonic()
    for i,row in enumerate(rows):
        file=dest/'samples'/(row['sample_id']+'.json')
        if file.exists():
            record=read(file);assert record['evaluation_signature']==signature
        else:
            data=store.get(row);results={}
            for key,model in models.items():
                score,windows,peak=score_sample(model,data,True)
                results[key]=dict(score=score,threshold=thresholds[key],prediction=predict(score,thresholds[key]),peak_window=peak,
                    windows=[dict(start=w['start'],end=w['end'],score=v,usable=w['usable']) for w,v in zip(data['windows'],windows)],unknown_reasons=data['unknown_reasons'])
            record=dict(sample_id=row['sample_id'],path=row['path'],source_sha256=row['sha256'],label=row['label'],split=row['split'],group=row['group'],
                origin=row.get('origin','unspecified'),source_label=row.get('source_label'),scope_basis=row.get('scope_basis'),
                source_review_status=row.get('source_review_status','inherited original manifest; not new human confirmation'),
                diagnostics=data['diagnostics'],results=results,evaluation_signature=signature)
            write(file,record)
        records.append(record)
        if (i+1)%25==0 or i+1==len(rows):
            progress=dict(phase=description,dataset=name,completed=i+1,total=len(rows),models_completed=12,models_total=12,seconds=time.monotonic()-started)
            write(OUT/'progress.json',progress);print(progress,flush=True)
    result=summarize(records);result['description']=description;write(dest/'summary.json',result)
    index=[]
    for row in records:
        for key,r in row['results'].items():
            p=r['prediction'];category='unknown' if p==-1 else 'false_positive' if p==1 and row['label']==0 else 'false_negative' if p==0 and row['label']==1 else 'correct'
            index.append(dict(sample_id=row['sample_id'],path=row['path'],label=row['label'],model=key,category=category,score=r['score'],threshold=r['threshold'],
                unknown_reasons=r['unknown_reasons'],positive_unknown_is_miss=bool(p==-1 and row['label']==1)))
    write(dest/'prediction_index.json',dict(rows=index));write(dest/'error_unknown_index.json',dict(rows=[r for r in index if r['category']!='correct']))
    return result

def run():
    setup_runtime();sealed=freeze_selections();signature=digest(sealed)
    for name,manifest in manifests().items():
        models={};thresholds={}
        for key in KEYS:
            arm,seed=key.split('_');folder=OUT/name/arm/f'seed_{seed}'
            saved=torch.load(folder/'selected_best.pt',map_location='cpu',weights_only=True)
            model=STGCNPPActionModel(manifest['task']);model.load_state_dict(saved['state_dict']);models[key]=model.cuda().eval();thresholds[key]=saved['threshold']
        # Re-evaluate the entire validation population before any heldout access.
        validation=[r for r in manifest['rows'] if r['split']=='validation'];store=ABStore(name)
        for key,model in models.items():
            arm,seed=key.split('_');expected=read(OUT/name/arm/f'seed_{seed}'/'best_validation_predictions.json')['rows']
            actual=evaluate(model,validation,store,dict(path=OUT/'progress.json',info=dict(dataset=name,model=key,models_completed=12,models_total=12)))
            assert [r['sample_id'] for r in expected]==[r['sample_id'] for r in validation]
            if actual!=[r['score'] for r in expected]:raise ValueError('Full validation inference parity failed: '+name+'/'+key)
        write(OUT/'verification'/f'validation_parity_{name}.json',dict(all_scores_exact=True,samples=len(validation),models=KEYS,test_used=False))
        evaluate_population(name,[r for r in manifest['rows'] if r['split']=='test'],models,thresholds,signature,OUT/'evaluations'/name/'retained_test','retained_test_with_prior_evaluation_history')
        regression='gmd' if name=='fallvision' else 'tnue';original=read(CACHE/'manifests'/f'{regression}.json')
        for split in ['train','validation','test']:
            rows=[r for r in original['rows'] if r['split']==split]
            if rows:evaluate_population(regression,rows,models,thresholds,signature,OUT/'evaluations'/name/'regression'/split,
                'original_'+regression+'_'+split+'_regression_only_prior_training_history'+('_provisional_labels' if regression=='tnue' else ''))
        del models;torch.cuda.empty_cache()
    write(OUT/'evaluations/complete.json',dict(status='complete',selection_signature=signature,all_12_models_selected_before_test=True))
    print('EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':run()
