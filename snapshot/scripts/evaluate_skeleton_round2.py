"""Locked second-round holdout and prior-training regression comparisons, offline."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,copy,os,time,math
import numpy as np
import torch
from scripts.skeleton_common import ROOT,CACHE,POSE,PROTOCOL,read,sha,digest,seal,offline
from scripts.skeleton_io import write
from scripts.skeleton_round2 import OUT,MANIFESTS,require_source_seal,choose_threshold,from_confusion
from scripts.train_skeleton_round2 import SampleStore,make_rgb,window_logit
from scripts.evaluate_skeleton_comparison import evaluate_sample,observation_quality
from backend.vision.skeleton_actions import SkeletonActionModel,prepare_skeleton

OLD=ROOT/'results/video_events/skeleton_comparison'
KEYS=[f'{m}_{s}' for m in ['rgb','skeleton'] for s in [42,43,44]]
VARIANTS=['round1_original_threshold','round1_validation_recalibrated','round2']

class EvaluationRgbStore(SampleStore):
    """Read matching original regression backbone caches without re-extracting them."""
    def __init__(self,name,rows):
        super().__init__(name,rows,'rgb',allow_feature_build=True)
        self.legacy_reused={}
        self.legacy_signature=digest(dict(pretrained=self.pretrain_sha,protocol=PROTOCOL,version=1))
        inventory=read(OUT/'audit/first_round_inventory.json')['files']
        producer=ROOT/'scripts/train_skeleton_comparison.py'
        prior=next(r for r in inventory if Path(r['path']).resolve()==producer.resolve())
        if sha(producer)!=prior['sha256']:raise ValueError('Original RGB cache producer changed')

    def get(self,row):
        legacy=CACHE/self.name/'rgb_features'/(row['sample_id']+'.pt')
        if self.name in ['gmd','tnue'] and legacy.exists():
            saved=torch.load(legacy,map_location='cpu',weights_only=True)
            if saved['signature']!=self.legacy_signature or saved['source_signature']!=self.expected[row['sample_id']]:
                raise ValueError('Legacy frozen RGB cache provenance mismatch')
            self.legacy_reused[row['sample_id']]=dict(path=str(legacy),sha256=sha(legacy),extractor_format_version=1)
            return saved
        return super().get(row)

def evaluate_new_sample(row,cache,models,thresholds,stages,rgb_store):
    """Use exactly the CPU pose preparation / cached RGB / fp16 score training path."""
    records={key:[] for key in models}
    if cache is not None:
        poses=[prepare_skeleton(clip,'cpu') for clip in cache['clips']]
        rgb=rgb_store.get(row)
        with torch.inference_mode(),torch.autocast('cuda'):
            for index,clip in enumerate(cache['clips']):
                quality=observation_quality(clip,poses[index],models['skeleton_42'].task)
                for key,model in models.items():
                    modality=key.split('_')[0];data=poses if modality=='skeleton' else rgb
                    logit=window_logit(model,data,modality,stages[key],index)
                    score=float(torch.tensor(float(logit),device='cuda',dtype=torch.float16).sigmoid()) if logit is not None else None
                    if score is not None and not math.isfinite(score):raise ValueError('Nonfinite evaluation score: '+row['sample_id']+'/'+key)
                    records[key].append(dict(start=clip['start'],end=clip['end'],score=score,quality=quality))
    results={}
    for key,items in records.items():
        valid=[(i,r['score']) for i,r in enumerate(items) if r['score'] is not None]
        peak=max(valid,key=lambda x:x[1]) if valid else None;score=peak[1] if peak else None
        results[key]=dict(score=score,prediction=predict(score,thresholds[key]),threshold=thresholds[key],
            peak_window=peak[0] if peak else None,windows=items,
            unknown_reasons=sorted({r['quality']['skeleton_unknown_reason'] for r in items if r['quality']['skeleton_unknown_reason']}) if score is None and items else (['source_extraction_failed'] if cache is None else []))
    shape=cache['source_shape'] if cache else [0,0];quality=[c['quality'] for c in cache['clips']] if cache else []
    ratio=float(np.median([q['median_person_height']/max(1,shape[0]) for q in quality])) if quality else 0
    coverage=float(np.mean([q['frame_coverage'] for q in quality])) if quality else 0
    simultaneous=max([int((c['boxes'][...,2]>c['boxes'][...,0]).sum(0).max()) for c in cache['clips']] or [0]) if cache else 0
    return dict(sample_id=row['sample_id'],path=row['path'],label=row['label'],origin=row['origin'],group=row['group'],
        scope_basis=row['scope_basis'],source_sha256=row['sha256'],results=results,
        diagnostics=dict(pose_frame_coverage=coverage,person_height_ratio=ratio,
            person_size='small' if ratio<.15 else 'medium' if ratio<.35 else 'large',
            pose_visibility_proxy='low' if coverage<.5 else 'partial' if coverage<.9 else 'high',
            simultaneous_detections=simultaneous,crowd='multiple' if simultaneous>=3 else 'zero_to_two'))

def metric(rows,key,variant):
    cm=np.zeros((2,3),dtype=int)
    for r in rows:
        pred=r['comparisons'][variant][key]['prediction']
        cm[r['label'],2 if pred==-1 else pred]+=1
    return from_confusion(cm)

def summary(rows):
    result=dict(samples=len(rows),sample_ids_digest=digest([r['sample_id'] for r in rows]),variants={})
    assert len(rows)==len({r['sample_id'] for r in rows})
    for variant in VARIANTS:
        per={k:metric(rows,k,variant) for k in KEYS};agg={};strata={}
        for mod in ['rgb','skeleton']:
            values=[per[f'{mod}_{s}'] for s in [42,43,44]]
            agg[mod]={}
            for attr in ['precision','recall','f1','macro_f1','normal_false_positive_rate','coverage','confusion_matrix']:
                a=np.asarray([v[attr] for v in values],dtype=float)
                agg[mod][attr]=dict(mean=a.mean(0).tolist(),std=a.std(0,ddof=1).tolist())
        for attr in ['origin','person_size','pose_visibility_proxy','crowd']:
            values=sorted({r['origin'] if attr=='origin' else r['diagnostics'][attr] for r in rows})
            strata[attr]={v:{k:metric([r for r in rows if (r['origin'] if attr=='origin' else r['diagnostics'][attr])==v],k,variant) for k in KEYS} for v in values}
        result['variants'][variant]=dict(per_seed=per,aggregate=agg,strata=strata)
    result['interpretation']='Unknowns remain in all denominators; positive unknowns are missed events. Size/visibility/crowd are pose-derived proxies, not human annotations. Standard deviation uses ddof=1 across three seeds.'
    result['same_denominators_all_models']=all(m['samples']==len(rows) for v in result['variants'].values() for m in v['per_seed'].values())
    return result

def predict(score,threshold):return -1 if score is None else int(score>=threshold)

def recalibrate(record,thresholds):
    out=copy.deepcopy(record['results'])
    for key,r in out.items():
        r['threshold']=thresholds[key];r['prediction']=predict(r['score'],thresholds[key])
    return out

def bundle(row,new,old,thresholds,signature):
    assert row['sample_id']==new['sample_id']==old['sample_id']
    assert row['sha256']==new['source_sha256']==old['source_sha256']
    assert row['label']==new['label']==old['label']
    for key in KEYS:
        for result in [new['results'][key],old['results'][key]]:
            score=result['score']
            assert score is None or (math.isfinite(score) and 0<=score<=1)
            assert result['prediction']==predict(score,result['threshold'])
        if (new['results'][key]['score'] is None)!=(old['results'][key]['score'] is None):
            raise ValueError('Fixed input eligibility changed between rounds: '+row['sample_id']+'/'+key)
    result={k:v for k,v in new.items() if k!='results'}
    result.update(split=row['split'],source_label=row.get('source_label'),source_review_status=row.get('source_review_status','first_round_manifest'),
        evaluation_signature=signature,comparisons=dict(round1_original_threshold=old['results'],round1_validation_recalibrated=recalibrate(old,thresholds),round2=new['results']))
    return result

def checkpoints(name,task,new=True):
    oldname='gmd' if name=='fallvision' else 'tnue';models={};thresholds={};hashes={}
    for key in KEYS:
        mod,seed=key.split('_');path=(OUT/name if new else OLD/oldname)/mod/f'seed_{seed}'/'selected_best.pt'
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model=make_rgb(False) if mod=='rgb' else SkeletonActionModel(task)
        model.load_state_dict(saved['state_dict']);models[key]=model.cuda().eval()
        thresholds[key]=saved['threshold'];hashes[key]=sha(path)
    reference=models['rgb_42'].state_dict()
    for seed in [43,44]:
        current=models[f'rgb_{seed}'].state_dict()
        assert all(torch.equal(v,current[k]) for k,v in reference.items() if k.startswith(('stem.','layer1.','layer2.','layer3.')))
    return models,thresholds,hashes

def locked_config(name,manifest):
    """Called before holdout inference; requires all six validation-only selections."""
    oldname='gmd' if name=='fallvision' else 'tnue'
    original_config=read(OLD/'evaluations'/oldname/'config.json')
    assert original_config['protocol']==PROTOCOL and original_config['pose_sha256']==sha(POSE)
    selected={};oldhash={};original_thresholds={};calibration={};valrows=[r for r in manifest['rows'] if r['split']=='validation']
    cache={}
    for row in valrows:
        record=read(OLD/'evaluations'/oldname/'samples'/(row['sample_id']+'.json'))
        assert record['source_sha256']==row['sha256'] and record['label']==row['label']
        assert record['evaluation_signature']==digest(original_config)
        cache[row['sample_id']]=record
    for key in KEYS:
        mod,seed=key.split('_');folder=OUT/name/mod/f'seed_{seed}';s=read(folder/'selection.json')
        assert s['status']=='complete' and s['test_accessed'] is False
        assert s['sha256']==sha(folder/'selected_best.pt')
        config=read(folder/'config.json')
        assert config['manifest_sha256']==sha(MANIFESTS/f'{name}.json') and s['config_signature']==digest(config)
        assert s['epochs_trained']<=50 and s['validation']['normal_false_positive_rate']<=.05
        selected[key]=s
        path=OLD/oldname/mod/f'seed_{seed}'/'selected_best.pt'
        oldhash[key]=sha(path)
        assert oldhash[key]==original_config['checkpoint_hashes'][key]
        original_thresholds[key]=original_config['thresholds'][key]
        vals=[cache[r['sample_id']]['results'][key]['score'] for r in valrows]
        threshold,met=choose_threshold([r['label'] for r in valrows],vals)
        calibration[key]=dict(threshold=threshold,validation=met,source='new validation only; fixed first-round weights',
            rows=[dict(sample_id=r['sample_id'],label=r['label'],score=v) for r,v in zip(valrows,vals)])
    config=dict(dataset=name,manifest_sha256=sha(MANIFESTS/f'{name}.json'),source_seal_sha256=sha(MANIFESTS/f'{name}.seal.json'),
        pose_sha256=sha(POSE),protocol=PROTOCOL,old_checkpoint_hashes=oldhash,old_original_thresholds=original_thresholds,
        old_evaluation_config_sha256=sha(OLD/'evaluations'/oldname/'config.json'),selected_models=selected,
        old_calibration=calibration,regression_manifest_sha256=sha(CACHE/'manifests'/f'{oldname}.json'),
        code_sha256={p:sha(ROOT/p) for p in ['scripts/evaluate_skeleton_round2.py','scripts/evaluate_skeleton_comparison.py','scripts/train_skeleton_round2.py','scripts/skeleton_round2.py']},
        selection_frozen_before_test=True,test_description='Repartitioned retained test previously evaluated in round one; not a new blind external test',
        numerical_note='Round two uses CPU pose preparation and explicit fp16 logit/sigmoid scores, matching its training validation path. First-round cached/native scores are preserved; its fight temporal pooling returned fp32. Same threshold selection rule, separate native scoring pipelines. See NUMERICAL_NOTES.md.',
        regression_description='GMDCSA-24/TNUE retain original train/validation/test splits and first-round training history; TNUE provisional labels remain provisional. No round-two training or selection on regression data.')
    seal(OUT/'evaluations'/name/'config.json',config)
    return config,original_config

def parity_on_validation(name,manifest,models,thresholds,store,stages,rgb_store):
    # Fixed first six validation rows, independent of scores or test behavior.
    rows=[r for r in manifest['rows'] if r['split']=='validation'][:6];expected={}
    for key in KEYS:
        mod,seed=key.split('_');folder=OUT/name/mod/f'seed_{seed}';s=read(folder/'selection.json')
        pred=read(folder/(s['selected_stage']+'_validation_predictions.json'))
        expected[key]={r['sample_id']:r['score'] for r in pred['rows']}
    checks=[]
    for row in rows:
        result=evaluate_new_sample(row,store.raw(row),models,thresholds,stages,rgb_store)
        for key in KEYS:
            a=result['results'][key]['score'];b=expected[key][row['sample_id']]
            if a!=b:raise ValueError(f'Train/inference parity failed on validation {name}/{key}/{row["sample_id"]}: {a} != {b}')
        checks.append(row['sample_id'])
    write(OUT/'evaluations'/name/'validation_inference_parity.json',dict(exact_scores_equal=True,samples=checks,model_keys=KEYS,test_used=False))

def evaluate_holdout(name,manifest,config,original_config,models,thresholds):
    oldname='gmd' if name=='fallvision' else 'tnue';dest=OUT/'evaluations'/name;signature=digest(config)
    store=SampleStore(name,manifest['rows'],'skeleton');store.verify_sources()
    rgb_store=EvaluationRgbStore(name,manifest['rows'])
    stages={k:v['selected_stage'] for k,v in config['selected_models'].items()}
    parity_on_validation(name,manifest,models,thresholds,store,stages,rgb_store)
    recal={k:v['threshold'] for k,v in config['old_calibration'].items()}
    rows=[r for r in manifest['rows'] if r['split']=='test'];records=[];started=time.monotonic()
    for i,row in enumerate(rows):
        path=dest/'retained_test/samples'/(row['sample_id']+'.json')
        if path.exists():
            record=read(path)
            assert record['evaluation_signature']==signature
        else:
            old=read(OLD/'evaluations'/oldname/'samples'/(row['sample_id']+'.json'))
            assert old['evaluation_signature']==digest(original_config)
            new=evaluate_new_sample(row,store.raw(row),models,thresholds,stages,rgb_store)
            record=bundle(row,new,old,recal,signature);write(path,record)
        records.append(record)
        if (i+1)%25==0 or i+1==len(rows):
            write(dest/'progress.json',dict(phase='retained_test',dataset=name,completed=i+1,total=len(rows),elapsed_seconds=time.monotonic()-started))
            print(name,'retained test',i+1,'/',len(rows),flush=True)
    result=summary(records);result['config_signature']=signature
    result['test_description']=config['test_description']
    result['test_fpr_target_met']={v:{k:m['normal_false_positive_rate']<=.05 for k,m in d['per_seed'].items()} for v,d in result['variants'].items()}
    result['test_policy']='Test FPR may exceed 5%; all thresholds remain fixed from validation. No test-driven retuning.'
    write(dest/'retained_test/summary.json',result);indices(records,dest/'retained_test')
    rgb_store.close_extractor()

def indices(records,dest):
    items=[]
    for row in records:
        for variant in VARIANTS:
            for key,r in row['comparisons'][variant].items():
                p=r['prediction'];category='unknown' if p==-1 else 'false_positive' if row['label']==0 and p==1 else 'false_negative' if row['label']==1 and p==0 else 'correct'
                items.append(dict(sample_id=row['sample_id'],path=row['path'],label=row['label'],model=key,variant=variant,score=r['score'],threshold=r['threshold'],prediction=p,category=category,
                    positive_unknown_is_miss=bool(p==-1 and row['label']==1),unknown_reasons=r.get('unknown_reasons',[])))
    write(dest/'prediction_index.json',dict(rows=items))
    write(dest/'error_unknown_index.json',dict(rows=[r for r in items if r['category']!='correct']))

def evaluate_regression(name,config,newmodels,newthresholds):
    oldname='gmd' if name=='fallvision' else 'tnue';task='fall' if name=='fallvision' else 'fight'
    manifest=read(CACHE/'manifests'/f'{oldname}.json');rows=manifest['rows'];dest=OUT/'evaluations'/name/'regression';signature=digest(config)
    store=SampleStore(oldname,rows,'skeleton');store.verify_sources()
    rgb_store=EvaluationRgbStore(oldname,rows)
    stages={k:v['selected_stage'] for k,v in config['selected_models'].items()}
    oldmodels,oldthresholds,oldhash=checkpoints(name,task,new=False)
    assert oldhash==config['old_checkpoint_hashes'] and oldthresholds==config['old_original_thresholds']
    recal={k:v['threshold'] for k,v in config['old_calibration'].items()};records=[]
    for i,row in enumerate(rows):
        path=dest/'samples'/(row['sample_id']+'.json')
        if path.exists():
            record=read(path);assert record['evaluation_signature']==signature
            rgb_store.get(row)  # Re-verify and retain full feature provenance on resume.
        else:
            raw=store.raw(row);old=evaluate_sample(row,raw,oldmodels,oldthresholds);new=evaluate_new_sample(row,raw,newmodels,newthresholds,stages,rgb_store)
            record=bundle(row,new,old,recal,signature);write(path,record)
        records.append(record)
        if (i+1)%25==0 or i+1==len(rows):print(name,oldname,'regression',i+1,'/',len(rows),flush=True)
    result=dict(dataset=oldname,description=config['regression_description'],source_manifest_sha256=sha(CACHE/'manifests'/f'{oldname}.json'),
        original_label_status=manifest.get('label_status'),original_limitation=manifest.get('limitation'),by_original_split={})
    result['original_frozen_rgb_feature_reuse']=dict(count=len(rgb_store.legacy_reused),records=rgb_store.legacy_reused,
        method='Original extractor format version1; unchanged original producer hash; exact local Kinetics hash/protocol/source signatures. Read original tensors directly, without overwriting or repeating their extraction.')
    for split in sorted({r['split'] for r in rows}):
        subset=[r for r in records if r['split']==split];result['by_original_split'][split]=summary(subset)
    write(dest/'summary.json',result);indices(records,dest)
    rgb_store.close_extractor()

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['fallvision','vfd'],required=True);a=p.parse_args()
    offline();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    manifest=require_source_seal(MANIFESTS/f'{a.dataset}.json')
    config,original=locked_config(a.dataset,manifest)
    models,thresholds,hashes=checkpoints(a.dataset,manifest['task'])
    assert all(hashes[k]==config['selected_models'][k]['sha256'] for k in KEYS)
    evaluate_holdout(a.dataset,manifest,config,original,models,thresholds)
    evaluate_regression(a.dataset,config,models,thresholds)
    write(OUT/'evaluations'/a.dataset/'progress.json',dict(phase='complete',dataset=a.dataset,retained_test_and_regression_complete=True))
    print('EVALUATION COMPLETE',a.dataset,flush=True)

if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');main()
