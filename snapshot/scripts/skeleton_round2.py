"""Round-two validation policy and fail-closed source seal checks.

Deliberately separate from skeleton_common: first-round code hashes and threshold
behavior must remain reproducible. No training starts through this module.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from fractions import Fraction
import math
from pathlib import Path
import numpy as np
from scripts.skeleton_common import CACHE, ROOT, digest, isolation, read, sha

MANIFESTS = CACHE / 'round2/manifests'
OUT = ROOT / 'results/video_events/skeleton_comparison_round2'
POLICY = dict(version=2, datasets=['fallvision','vfd'], split_seed=42,
    seeds=[42,43,44], max_total_epochs=50,
    stage_epoch_limits={'rgb':{'baseline':10,'finetune':40},'skeleton':{'baseline':50}},
    patience=8, normal_false_positive_limit=.05,
    selection='validation event recall, lower normal false positive proportion, macro F1, higher threshold; earlier epoch/stage on complete tie',
    test_access='after all model/epoch/stage/threshold decisions are sealed',
    initialization={'rgb':'local Kinetics R3D-18','skeleton':'random'},
    regression_only=['gmd','tnue'],unknown='retained in total and true-class recall denominators',
    network='disabled; no automatic downloads',
    test_description='repartitioned retained test with first-round evaluation history')


def from_confusion(cm):
    cm=np.asarray(cm,dtype=np.int64)
    tp=np.array([cm[0,0],cm[1,1]],dtype=float)
    precision=tp/np.maximum(1,cm[:,:2].sum(0))
    recall=tp/np.maximum(1,cm.sum(1))
    f1=2*precision*recall/np.maximum(1e-12,precision+recall)
    return dict(samples=int(cm.sum()),confusion_matrix=cm.tolist(),
        prediction_columns=['normal','event','unknown'],precision=precision.tolist(),
        recall=recall.tolist(),f1=f1.tolist(),macro_f1=float(f1.mean()),
        accuracy=float(tp.sum()/max(1,cm.sum())),coverage=float(cm[:,:2].sum()/max(1,cm.sum())),
        normal_false_positive_rate=float(cm[0,1]/max(1,cm[0].sum())))


def selection_key(metrics):
    return (metrics['recall'][1],-metrics['normal_false_positive_rate'],metrics['macro_f1'])


def choose_threshold(truth,scores,limit=.05):
    """All >= decision partitions, including the finite no-event endpoint, O(n log n).

    Only call with validation samples. Unknowns cannot be converted to normal.
    Require both source classes rather than interpreting an empty denominator as 0.
    """
    if len(truth)!=len(scores) or set(truth)!={0,1}:
        raise ValueError('Validation requires equal-length labels/scores and both classes')
    if not math.isfinite(limit) or not 0<=limit<=1:
        raise ValueError('Invalid normal false-positive limit')
    cm=np.zeros((2,3),dtype=np.int64)
    changes=defaultdict(lambda:np.zeros(2,dtype=np.int64))
    for label,score in zip(truth,scores):
        if score is None:
            cm[int(label),2]+=1
        else:
            value=float(score)
            if not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('Scores must be finite sigmoid probabilities or None')
            cm[int(label),0]+=1
            changes[value][int(label)]+=1
    endpoint=float(np.nextafter(max(changes,default=1.),np.inf))
    best=from_confusion(cm);threshold=endpoint
    best_key=(*selection_key(best),threshold)
    budget=Fraction(str(limit))*int(cm[0].sum())
    considered=1;feasible=1
    for value in sorted(changes,reverse=True):
        counts=changes[value];cm[:,0]-=counts;cm[:,1]+=counts
        considered+=1
        if int(cm[0,1])>budget:
            continue
        feasible+=1;current=from_confusion(cm)
        key=(*selection_key(current),value)
        if key>best_key:
            best_key=key;threshold=value;best=current
    best.update(effective_detection=best['recall'][1]>0,
        detection_status='effective_on_validation' if best['recall'][1]>0 else '未获得有效检测能力',
        thresholds_examined=considered,feasible_thresholds=feasible,
        normal_false_positive_limit=limit)
    return threshold,best


def require_source_seal(path):
    """A draft or unresolved source relation cannot authorize model training."""
    path=Path(path)
    manifest=read(path)
    if manifest.get('status')!='sealed' or manifest.get('training_allowed') is not True:
        raise ValueError('Source manifest is not sealed/training-ready; source review must finish first')
    if manifest.get('split_seed')!=42 or manifest.get('dataset_key') not in POLICY['datasets']:
        raise ValueError('Unexpected dataset or split seed')
    rows=manifest['rows'];isolation(rows)
    for split in ('train','validation','test'):
        if {r['label'] for r in rows if r['split']==split}!={0,1}:
            raise ValueError(f'{split} must contain both classes')
    for row in rows:
        if row.get('source_review_status') not in ('AI_reviewed','human_confirmed','conservative_merged'):
            raise ValueError('Unreviewed source row')
        if not row.get('source_evidence') or row.get('new_source_group')!=row['group']:
            raise ValueError('Missing source evidence/group mapping')
        if row['label']!=row['original_label']:
            raise ValueError('Round-two source labels changed without a separately reviewed exclusion')
    seal_path=path.with_suffix('.seal.json')
    seal=read(seal_path)
    if seal['manifest_sha256']!=sha(path) or seal['row_split_digest']!=digest([(r['sample_id'],r['label'],r['group'],r['split']) for r in rows]):
        raise ValueError('Manifest seal mismatch')
    if seal.get('unresolved_source_relations')!=0 or seal.get('cross_split_duplicate_candidates')!=0:
        raise ValueError('Unresolved source or duplicate relationships')
    return manifest
