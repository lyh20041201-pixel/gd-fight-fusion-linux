"""Seal the A/B plan, inventory prior outputs, and convert matching pose caches."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.stgcnpp_ab import *

def main():
    setup_runtime();ms=manifests();OUT.mkdir(parents=True,exist_ok=True)
    plan=dict(policy=POLICY,models=[dict(dataset=n,arm=a,seed=s) for n in POLICY['datasets'] for s in POLICY['seeds'] for a in POLICY['arms']],
        manifests={n:dict(path=str(MANIFESTS/f'{n}.json'),sha256=sha(MANIFESTS/f'{n}.json'),split_counts=m['split_counts'],limitations=m['limitation']) for n,m in ms.items()},
        code_hashes={f:sha(ROOT/f) for f in CODE},download_receipt=read(OUT/'download_receipt.json'),
        experiment_description='Paired backbone-initialization comparison, same task-specific training budget. Not total-compute matched: B includes prior NTU60 pretraining.',
        rgb='Existing second-round RGB results unchanged; A/B concerns skeleton action models only')
    if (OUT/'experiment_plan.json').exists():
        draft=read(OUT/'experiment_plan.json')
        assert {k:v for k,v in draft.items() if k!='code_hashes'}=={k:v for k,v in plan.items() if k!='code_hashes'}
        # Keep the original preflight draft; the queue seals updated code hashes
        # in experiment_plan_frozen.json after all implementation checks pass.
    else:seal(OUT/'experiment_plan.json',plan)
    inventory_path=OUT/'audit/protected_outputs.json'
    if not inventory_path.exists():
        paths=[]
        for folder in ['results/video_events/skeleton_comparison','results/video_events/skeleton_comparison_round2','datasets/video_events/skeleton_rebuild/manifests','datasets/video_events/skeleton_rebuild/round2/manifests']:
            paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
        inventory=[]
        for i,p in enumerate(sorted(set(paths))):
            inventory.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
            if (i+1)%500==0:print('PROTECT_EXISTING_OUTPUTS',i+1,'/',len(paths),flush=True)
        seal(inventory_path,dict(files=inventory))
    for name,m in ms.items():prepare_dataset(name,m)
    for name in ['gmd','tnue']:prepare_dataset(name,read(CACHE/'manifests'/f'{name}.json'))
    # Fixed eligibility checks from the already completed source audit, not predictions.
    expected={'fallvision':{'train':{'0':31,'1':2},'validation':{'0':8,'1':0},'test':{'0':0,'1':0}},
              'vfd':{'train':{'0':254,'1':249},'validation':{'0':9,'1':9},'test':{'0':32,'1':18}}}
    for name,wanted in expected.items():
        actual=read(OUT/'audit'/f'inputs_{name}.json')['unknown_counts']
        if actual!=wanted:raise ValueError(f'Input eligibility changed: {name}: {actual} != {wanted}')
    write(OUT/'audit/preparation_complete.json',dict(status='complete',feature_signature=feature_signature(),eligibility_matches_round2=True,datasets=['fallvision','vfd','gmd','tnue']))
    print('PREPARATION_COMPLETE',flush=True)

if __name__=='__main__':main()
