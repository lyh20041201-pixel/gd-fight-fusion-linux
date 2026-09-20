"""Reconstruct every selection from validation history; never inspect test scores."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from scripts.skeleton_common import read,sha,digest
from scripts.skeleton_round2 import OUT,MANIFESTS,POLICY,selection_key,choose_threshold,require_source_seal
from scripts.skeleton_io import write

def verify(completed_only=False):
    verified=[]
    for name in ['fallvision','vfd']:
        manifest=require_source_seal(MANIFESTS/f'{name}.json')
        train=[r for r in manifest['rows'] if r['split']=='train'];validation=[r for r in manifest['rows'] if r['split']=='validation']
        for mod in ['rgb','skeleton']:
            for seed in [42,43,44]:
                base=OUT/name/mod/f'seed_{seed}'
                if completed_only and not (base/'selection.json').exists():continue
                selected=read(base/'selection.json');run=read(base/'run_record.json');config=read(base/'config.json')
                assert selected['status']==run['status']=='complete'
                assert selected['config_signature']==run['config_signature']==digest(config)
                assert config['manifest_sha256']==sha(MANIFESTS/f'{name}.json')
                assert selected['sha256']==sha(base/'selected_best.pt')
                history=[];stages={}
                for stage,limit in POLICY['stage_epoch_limits'][mod].items():
                    h=run['histories'][stage];assert 0<len(h)<=limit
                    best=max(h,key=lambda r:selection_key(r['validation']))
                    assert len(h)==limit or len(h)-best['epoch']==POLICY['patience']
                    for r in h:
                        assert r['used_training_segments']+len(r['unusable_training_segments'])==len(train)
                        assert len(r['unusable_training_segments'])==len(set(r['unusable_training_segments']))
                        assert r['validation']['samples']==len(validation)
                        assert r['validation']['normal_false_positive_rate']<=.05
                    stages[stage]=dict(epochs=len(h),limit=limit,best_epoch=best['epoch'])
                    history+=h
                assert selected['epochs_trained']==len(history)==run['epochs_total']<=50
                assert [r['cumulative_epoch'] for r in history]==list(range(1,len(history)+1))
                best=max(history,key=lambda r:selection_key(r['validation']))
                assert (selected['selected_stage'],selected['epoch'],selected['threshold'])==(best['stage'],best['epoch'],best['threshold'])
                predictions=read(base/(selected['selected_stage']+'_validation_predictions.json'))['rows']
                assert [(r['sample_id'],r['label']) for r in predictions]==[(r['sample_id'],r['label']) for r in validation]
                threshold,metrics=choose_threshold([r['label'] for r in predictions],[r['score'] for r in predictions])
                assert threshold==selected['threshold'] and metrics==selected['validation']
                assert selected['test_accessed'] is False
                verified.append(dict(dataset=name,modality=mod,seed=seed,epochs_trained=len(history),stages=stages,
                    selected_stage=selected['selected_stage'],selected_cumulative_epoch=selected['cumulative_epoch'],
                    threshold=threshold,validation_effective_detection=metrics['effective_detection']))
    result=dict(status='passed',completed_models=len(verified),expected_models=12,validation_only=True,
        threshold_boundaries_reconstructed=True,early_stopping_and_stage_selection_reconstructed=True,unknown_training_denominators_verified=True,models=verified)
    if not completed_only:assert len(verified)==12
    write(OUT/'verification'/('selection_audit_partial.json' if completed_only else 'selection_audit_final.json'),result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--completed-only',action='store_true');a=p.parse_args()
    result=verify(a.completed_only);print('SELECTION AUDIT PASSED',result['completed_models'],'/',12,flush=True)
