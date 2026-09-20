"""One immutable prediction per external sample, shared frames and full denominators."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,time
import numpy as np,torch
from scripts.skeleton_common import *
from scripts.skeleton_io import write
from scripts.train_skeleton_comparison import make_rgb
from backend.vision.skeleton_actions import SkeletonActionModel,prepare_skeleton

def observation_quality(clip,pose,task):
    """Describe observed tracks and eligibility without reconstructing hidden people."""
    info=dict(clip['quality']);valid=clip['frame_valid'].numpy()
    times=np.asarray(clip['timestamps']);indices=np.asarray(clip['frame_indices'])
    spans=[]
    for tid,mask in zip(clip['track_ids'],valid):
        observed=times[mask];unique=len(np.unique(indices[mask]))
        spans.append(dict(track_id=tid,unique_valid_frames=unique,usable=unique>=8,
            first_observed=float(observed[0]) if len(observed) else None,
            last_observed=float(observed[-1]) if len(observed) else None,
            missing_sample_indices=np.flatnonzero(~mask).tolist(),
            observed_gaps_seconds=np.diff(observed).tolist(),
            partial_window=bool(mask.sum()<len(mask))))
    count=int(pose['features'].shape[0]);info['usable_tracks']=count
    info['track_observations']=spans
    pair_count=0
    if task=='fight' and count>=2:
        pairs=torch.triu_indices(count,count,offset=1,device=pose['valid'].device)
        shared=pose['valid'][pairs[0]] & pose['valid'][pairs[1]]
        pair_count=int((shared.sum(-1)>=8).sum())
    info['eligible_pairs']=pair_count if task=='fight' else None
    reason=None
    if count==0:reason='no_track_with_eight_unique_valid_frames'
    elif task=='fight' and count<2:reason='fewer_than_two_usable_tracks'
    elif task=='fight' and pair_count==0:reason='insufficient_simultaneous_pair_observations'
    info['skeleton_unknown_reason']=reason
    info['interpretation']='Missing observations and track boundaries may reflect occlusion, leaving frame or tracking failure; identities across gaps are not reconstructed.'
    return info

def evaluate_sample(row,cache,models,thresholds):
    records={key:[] for key in models}
    if cache is not None:
        for clip in cache['clips']:
            pose=prepare_skeleton(clip,'cuda')
            quality=observation_quality(clip,pose,models['skeleton_42'].task)
            with torch.inference_mode(),torch.autocast('cuda'):
                # Stem and layer1..3 are frozen identical Kinetics weights in every RGB seed.
                common=models['rgb_42'];x=rgb_tensor(clip['rgb']).unsqueeze(0).cuda()
                z=common.layer3(common.layer2(common.layer1(common.stem(x))))
                for key,model in models.items():
                    if key.startswith('rgb'):
                        logits=model.fc(model.avgpool(model.layer4(z)).flatten(1))
                        value=logits[0,1]-logits[0,0]
                    else:value=model(pose)
                    score=float(value.sigmoid()) if value is not None else None
                    records[key].append(dict(start=clip['start'],end=clip['end'],score=score,
                                              quality=quality))
    results={}
    for key,items in records.items():
        valid=[(i,r['score']) for i,r in enumerate(items) if r['score'] is not None]
        peak=max(valid,key=lambda x:x[1]) if valid else None
        score=peak[1] if peak else None
        results[key]=dict(score=score,prediction=-1 if score is None else int(score>=thresholds[key]),
                          threshold=thresholds[key],peak_window=peak[0] if peak else None,windows=items,
                          unknown_reasons=sorted({r['quality']['skeleton_unknown_reason'] for r in items
                              if r['quality']['skeleton_unknown_reason']}) if score is None and items else
                              (['source_extraction_failed'] if cache is None else []))
    shape=cache['source_shape'] if cache else [0,0]
    quality=[c['quality'] for c in cache['clips']] if cache else []
    ratio=float(np.median([q['median_person_height']/max(1,shape[0]) for q in quality])) if quality else 0
    coverage=float(np.mean([q['frame_coverage'] for q in quality])) if quality else 0
    simultaneous=max([int((c['boxes'][...,2]>c['boxes'][...,0]).sum(0).max()) for c in cache['clips']] or [0]) if cache else 0
    return dict(sample_id=row['sample_id'],path=row['path'],label=row['label'],origin=row['origin'],group=row['group'],
                scope_basis=row['scope_basis'],source_sha256=row['sha256'],results=results,
                diagnostics=dict(pose_frame_coverage=coverage,person_height_ratio=ratio,
                                 person_size='small' if ratio<.15 else 'medium' if ratio<.35 else 'large',
                                 pose_visibility_proxy='low' if coverage<.5 else 'partial' if coverage<.9 else 'high',
                                 simultaneous_detections=simultaneous,crowd='multiple' if simultaneous>=3 else 'zero_to_two'))

def summarize(records,keys,expected):
    if len(records)!=expected or len({r['sample_id'] for r in records})!=expected:raise ValueError('External coverage mismatch')
    metrics_by={};strata={}
    for key in keys:
        metrics_by[key]=metrics([r['label'] for r in records],[r['results'][key]['prediction'] for r in records])
        strata[key]={}
        for attr in ['origin','person_size','pose_visibility_proxy','crowd']:
            values=sorted({r[attr] if attr=='origin' else r['diagnostics'][attr] for r in records})
            strata[key][attr]={}
            for val in values:
                subset=[r for r in records if (r[attr] if attr=='origin' else r['diagnostics'][attr])==val]
                strata[key][attr][val]=metrics([r['label'] for r in subset],[r['results'][key]['prediction'] for r in subset])
    aggregate={}
    for modality in ['rgb','skeleton']:
        data=[metrics_by[f'{modality}_{seed}'] for seed in [42,43,44]]
        aggregate[modality]={k:dict(mean=float(np.mean([r[k] for r in data])),std=float(np.std([r[k] for r in data],ddof=1)))
                             for k in ['accuracy','macro_f1','coverage','normal_false_positive_rate']}
        for metric in ['precision','recall','f1']:
            vals=np.array([r[metric] for r in data]);aggregate[modality][metric]=dict(mean=vals.mean(0).tolist(),std=vals.std(0,ddof=1).tolist())
    return dict(status='complete',samples=expected,per_seed=metrics_by,aggregate=aggregate,strata=strata,
                interpretation='Pose visibility and size are automated diagnostics, not human occlusion annotations; subgroup metrics may have only one class.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['gmd','tnue'],required=True);a=p.parse_args()
    offline();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    ext='fallvision' if a.dataset=='gmd' else 'vfd';task='fall' if a.dataset=='gmd' else 'fight'
    manifest=read(CACHE/'manifests'/f'{ext}.json');rows=manifest['rows'];isolation(rows)
    extraction=read(OUT/f'extraction_{ext}.json')
    if extraction['completed']+len(extraction['failures'])!=len(rows):raise RuntimeError('External extraction has not covered all samples')
    failed={r['sample_id'] for r in extraction['failures']}
    models={};thresholds={};hashes={}
    for modality in ['rgb','skeleton']:
        for seed in [42,43,44]:
            path=OUT/a.dataset/modality/f'seed_{seed}'/'selected_best.pt';saved=torch.load(path,map_location='cpu',weights_only=True)
            model=make_rgb(False) if modality=='rgb' else SkeletonActionModel(task)
            model.load_state_dict(saved['state_dict']);key=f'{modality}_{seed}'
            models[key]=model.cuda().eval();thresholds[key]=saved['threshold'];hashes[key]=sha(path)
    reference=models['rgb_42'].state_dict()
    for seed in [43,44]:
        current=models[f'rgb_{seed}'].state_dict()
        assert all(torch.equal(v,current[k]) for k,v in reference.items() if k.startswith(('stem.','layer1.','layer2.','layer3.')))
    out=OUT/'evaluations'/a.dataset;out.mkdir(parents=True,exist_ok=True)
    config=dict(external_dataset=ext,manifest_sha256=sha(CACHE/'manifests'/f'{ext}.json'),checkpoint_hashes=hashes,
                thresholds=thresholds,protocol=PROTOCOL,evaluator_version=2,code_sha256=sha(__file__),pose_sha256=sha(POSE),
                label_limitation=manifest['limitation'])
    seal(out/'config.json',config);signature=digest(config);records=[];started=time.monotonic()
    for i,row in enumerate(rows):
        path=out/'samples'/(row['sample_id']+'.json')
        if path.exists():
            record=read(path)
            if record['evaluation_signature']!=signature:raise ValueError('Evaluation resume signature mismatch')
        else:
            cache_path=CACHE/ext/(row['sample_id']+'.pt')
            if not cache_path.exists() and row['sample_id'] not in failed:raise ValueError('Unaccounted missing source cache')
            cache=torch.load(cache_path,map_location='cpu',weights_only=True) if cache_path.exists() else None
            if cache and (cache['source_sha256']!=row['sha256'] or cache['protocol']!=PROTOCOL or cache['pose_sha256']!=config['pose_sha256']):raise ValueError('Cache provenance mismatch')
            record=evaluate_sample(row,cache,models,thresholds);record['evaluation_signature']=signature;write(path,record)
        records.append(record)
        if i%25==0 or i==len(rows)-1:
            write(out/'progress.json',dict(status='running',completed=i+1,expected=len(rows),elapsed_seconds=time.monotonic()-started))
            print(ext,'evaluated',i+1,'/',len(rows),flush=True)
    result=summarize(records,list(models),len(rows));result['config']=config
    write(out/'summary.json',result);write(out/'progress.json',dict(status='complete',completed=len(rows),expected=len(rows)))
    print('EXTERNAL COMPLETE',ext,result['aggregate'],flush=True)

if __name__=='__main__':main()
