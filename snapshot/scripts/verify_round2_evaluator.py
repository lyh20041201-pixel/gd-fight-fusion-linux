"""Integration check using fixed validation rows only, before test access."""
from pathlib import Path
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.skeleton_common import read,offline
from scripts.skeleton_round2 import OUT,MANIFESTS
from scripts.skeleton_io import write
from scripts.evaluate_skeleton_round2 import checkpoints,OLD,evaluate_sample,evaluate_new_sample,SampleStore,make_rgb
from backend.vision.skeleton_actions import SkeletonActionModel

def main():
    offline();torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    manifest=read(MANIFESTS/'vfd.json');rows=[r for r in manifest['rows'] if r['split']=='validation'][:6]
    store=SampleStore('vfd',rows,'skeleton');rgb_store=SampleStore('vfd',rows,'rgb');models={};thresholds={};expected={};stages={}
    rgb_folder=OUT/'vfd/rgb/seed_42';rgb_saved=torch.load(rgb_folder/'baseline_best.pt',map_location='cpu',weights_only=True)
    rgb_model=make_rgb(False);rgb_model.load_state_dict(rgb_saved['state_dict']);rgb_model=rgb_model.cuda().eval()
    rgb_expected={r['sample_id']:r['score'] for r in read(rgb_folder/'baseline_validation_predictions.json')['rows']}
    for seed in [42,43,44]:
        folder=OUT/'vfd/skeleton'/f'seed_{seed}';saved=torch.load(folder/'selected_best.pt',map_location='cpu',weights_only=True)
        model=SkeletonActionModel('fight');model.load_state_dict(saved['state_dict']);models[f'skeleton_{seed}']=model.cuda().eval();thresholds[f'skeleton_{seed}']=saved['threshold']
        vals=read(folder/(saved['stage']+'_validation_predictions.json'))['rows'];expected[f'skeleton_{seed}']={r['sample_id']:r['score'] for r in vals}
        stages[f'skeleton_{seed}']=saved['stage']
        models[f'rgb_{seed}']=rgb_model;thresholds[f'rgb_{seed}']=rgb_saved['threshold'];stages[f'rgb_{seed}']='baseline';expected[f'rgb_{seed}']=rgb_expected
    checks=[]
    for row in rows:
        result=evaluate_new_sample(row,store.raw(row),models,thresholds,stages,rgb_store)
        for key in models:
            ref=expected[key][row['sample_id']]
            value=result['results'][key]['score']
            if value!=ref:raise ValueError(f'Validation inference mismatch: {key} {row["sample_id"]} {value} != {ref}')
        checks.append(row['sample_id'])
    write(OUT/'verification/evaluator_integration.json',dict(status='passed',validation_only=True,sample_ids=checks,
        exact_new_skeleton_scores_match_selected_validation_predictions=True,exact_new_rgb_scores_match_baseline_validation_predictions=True,
        rgb_check_note='Baseline seed42 aliased for this pre-completion integration check; final evaluator separately requires all six selected models to pass validation parity.',model_keys=list(models)))
    print('EVALUATOR INTEGRATION PASSED',len(rows),'validation samples x',len(models),'models',flush=True)

if __name__=='__main__':os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');main()
