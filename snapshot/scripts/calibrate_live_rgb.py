"""Validation-only RGB ensemble selection; preserve all sealed experiments."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, time
import numpy as np
import torch
from scripts.skeleton_common import ROOT, CACHE, read, sha, offline
from scripts.skeleton_io import write
from scripts.skeleton_round2 import choose_threshold, selection_key, from_confusion
from scripts.train_skeleton_round2 import SampleStore, window_logit
from scripts.train_skeleton_comparison import make_rgb

OUT = ROOT / 'results/live_actions/rgb_ensemble_v1'
BASE = ROOT / 'results/video_events/skeleton_comparison_round2/vfd/rgb'

def aggregate(windows, method):
    values = np.asarray(windows, dtype=float)
    if values.size == 0:
        return None
    if method.startswith('single_'):
        return float(values[int(method.split('_')[1])].max())
    merged = values.mean(0) if method == 'mean' else np.median(values, axis=0)
    return float(merged.max())

def main():
    offline(); torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read(CACHE/'round2/manifests/vfd.json')
    rows = [r for r in manifest['rows'] if r['split'] == 'validation']
    store = SampleStore('vfd', rows, 'rgb')
    selections = [read(BASE/f'seed_{seed}/selection.json') for seed in (42,43,44)]
    models = []
    for seed, sel in zip((42,43,44), selections):
        path = BASE/f'seed_{seed}/selected_best.pt'
        assert sha(path) == sel['sha256']
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = make_rgb(False); model.load_state_dict(saved['state_dict'])
        models.append(model.cuda().eval())
    records = []
    with torch.inference_mode(), torch.autocast('cuda'):
        for i, row in enumerate(rows):
            data = store.get(row)
            windows = []
            for model, sel in zip(models, selections):
                values = [float(window_logit(model, data, 'rgb', sel['selected_stage'], j).sigmoid())
                          for j in range(len(data['layer3']))] if data else []
                windows.append(values)
            records.append(dict(sample_id=row['sample_id'], label=row['label'], windows=windows))
            if (i+1)%25 == 0: print('VALIDATION', i+1, len(rows), flush=True)
    write(OUT/'validation_windows.json', dict(rows=records))
    candidates=[]
    for method in ('single_0','single_1','single_2','mean','median'):
        scores=[aggregate(r['windows'], method) for r in records]
        threshold, metric=choose_threshold([r['label'] for r in records], scores)
        candidates.append(dict(method=method,threshold=threshold,validation=metric))
    # A complete tie prefers a single model, then the earlier listed method.
    selected=max(candidates,key=lambda r:selection_key(r['validation']))
    selection=dict(status='sealed',selected=selected,candidates=candidates,
        checkpoint_paths=[str(BASE/f'seed_{s}/selected_best.pt') for s in (42,43,44)],
        checkpoint_sha256=[s['sha256'] for s in selections],
        policy='Maximum over window-wise mean/median probabilities; validation FPR<=5%, recall, lower FPR, macro F1. Ties prefer single models. No test tuning.',
        selection_split='validation',selected_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        test_accessed_for_selection=False,code_sha256=sha(__file__))
    path=OUT/'selection.json'
    if path.exists():
        prior=read(path)
        assert prior['selected']==selected, 'Do not overwrite a different sealed selection'
        selection=prior
    else: write(path,selection)
    print('SEALED',json.dumps(selected),flush=True)
    # Heldout access follows selection; no decisions below may change selection.
    folder=ROOT/'results/video_events/skeleton_comparison_round2/evaluations/vfd/retained_test/samples'
    test=[]
    for row in manifest['rows']:
        if row['split']!='test': continue
        raw=read(folder/(row['sample_id']+'.json'))
        results=raw['comparisons']['round2'] if 'comparisons' in raw else raw['results']
        values=[[w['score'] for w in results[f'rgb_{s}']['windows']] for s in (42,43,44)]
        test.append(dict(sample_id=row['sample_id'],label=row['label'],windows=values))
    metrics={}
    for candidate in candidates:
        cm=np.zeros((2,3),dtype=int)
        for row in test:
            score=aggregate(row['windows'],candidate['method'])
            cm[row['label'],2 if score is None else int(score>=candidate['threshold'])]+=1
        metrics[candidate['method']]=from_confusion(cm)
    write(OUT/'heldout_evaluation.json',dict(selected=selected['method'],metrics=metrics,
        thresholds_retuned=False,test_has_prior_evaluation_history=True))
    print('HELDOUT',json.dumps(metrics[selected['method']]),flush=True)

if __name__=='__main__': main()
