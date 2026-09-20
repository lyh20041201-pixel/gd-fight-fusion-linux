"""Deterministic grouped split, preserving every known/potential source edge."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from collections import defaultdict,Counter
import numpy as np
from scripts.skeleton_common import CACHE,ROOT,PROTOCOL,read,write,sha,digest,seal,isolation,offline
from scripts.skeleton_round2 import MANIFESTS,OUT,require_source_seal
from scripts.audit_skeleton_round2 import youtube_id


def split_groups(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['group']].append(r)
    rng=np.random.default_rng(42);keys=sorted(groups);rng.shuffle(keys)
    keys.sort(key=lambda g:-len(groups[g]))
    totals=np.array([sum(r['label']==c for r in rows) for c in [0,1]])
    targets=np.array([.7,.15,.15])[:,None]*totals[None]
    current=np.zeros((3,2),dtype=int);assignment={}
    for key in keys:
        count=np.array([sum(r['label']==c for r in groups[key]) for c in [0,1]])
        options=[]
        for s in range(3):
            after=current.copy();after[s]+=count
            options.append((float(np.square((after-targets)/totals).sum()),s))
        chosen=min(options)[1];current[chosen]+=count
        assignment[key]=['train','validation','test'][chosen]
    if (current==0).any():raise ValueError('Grouped split cannot populate all three sets with both classes')
    return assignment


def publish(name,rows,exclusions,evidence,limitations):
    for row in rows:
        row['new_source_group']=row['group']
    assignment=split_groups(rows)
    for row in rows:row['split']=assignment[row['group']]
    isolation(rows)
    result=dict(status='sealed',training_allowed=True,dataset_key=name,
        dataset='FallVision' if name=='fallvision' else 'VFD-2000',
        task='fall' if name=='fallvision' else 'fight',labels=['normal','fall' if name=='fallvision' else 'fight'],
        split_seed=42,training_seeds=[42,43,44],target_ratios={'train':.7,'validation':.15,'test':.15},
        split_algorithm='largest source groups first; deterministic seeded tie order; minimize class-normalized squared count deviations',
        split_counts={s:dict(Counter(r['label'] for r in rows if r['split']==s)) for s in ['train','validation','test']},
        source_groups=len(assignment),rows=rows,exclusions=exclusions,
        parent_manifest=str(CACHE/'manifests'/f'{name}.json'),parent_manifest_sha256=sha(CACHE/'manifests'/f'{name}.json'),
        source_evidence=evidence,limitation=limitations,
        label_status='original source labels plus inherited scope exclusions; AI source grouping; no human identity confirmation',
        subject_independent=False,test_description='Repartitioned retained test with first-round evaluation history; not a new blind external test',
        sampling_protocol=PROTOCOL)
    path=MANIFESTS/f'{name}.json'
    seal(path,result)
    seal(path.with_suffix('.seal.json'),dict(manifest_sha256=sha(path),
        row_split_digest=digest([(r['sample_id'],r['label'],r['group'],r['split']) for r in rows]),
        unresolved_source_relations=0,cross_split_duplicate_candidates=0,
        interpretation='Zero uncontained documented source relations: ambiguous within-group links are allowed and explicitly retained. This is not proof that no undiscovered duplicate exists.',
        evidence_sha256={p:sha(p) for p in evidence}))
    require_source_seal(path)
    write(OUT/f'split_{name}.json',dict(status='sealed',samples=len(rows),counts=result['split_counts'],groups=len(assignment),
        fractions={s:sum(r['split']==s for r in rows)/len(rows) for s in ['train','validation','test']},limitations=limitations))
    print('SEALED',name,len(rows),result['split_counts'],len(assignment),'source groups',flush=True)


def vfd():
    source=read(CACHE/'manifests/vfd.json');draft=read(MANIFESTS/'vfd_source_review_draft.json')
    candidates=read(OUT/'source_review/vfd_cross_url_candidates.json')
    by={r['sample_id']:r for r in draft['rows']};rows=[]
    for original in source['rows']:
        r=dict(original);d=by[r['sample_id']]
        r.update(original_label=r['label'],original_group=r['group'],original_split=r['split'],
            group=d['proposed_source_group'],source_review_status='conservative_merged',human_confirmed=False,
            normalized_source_id=youtube_id(r['group']),
            source_evidence=str(OUT/'source_review/vfd_containment_decision.json'),
            source_audit_record=str(OUT/'audit/samples/vfd'/(r['sample_id']+'.json')))
        rows.append(r)
    group_by_url={r['normalized_source_id']:r['group'] for r in rows}
    for pair in candidates['candidates']:
        assert group_by_url[pair['source_a']]==group_by_url[pair['source_b']]
    p=OUT/'source_review/vfd_containment_decision.json'
    seal(p,dict(status='all_documented_candidate_relations_contained',reviewed_candidates=129,
        decision='Retain all 129 candidate edges, including visually unconfirmed links, and forbid splitting their entire URL connected components. No claim that all merged URLs show the same event.',
        candidate_file=str(OUT/'source_review/vfd_cross_url_candidates.json'),candidate_sha256=sha(OUT/'source_review/vfd_cross_url_candidates.json'),
        source_ids=168,components=78,uncontained_documented_relations=0,
        label_changes=0,training_relevance='A complete identity map within each component is unnecessary for split isolation once the entire component is inseparable.',
        limitation='Five-frame pHash plus AI visual review can miss transformations. All detected or uncertain links are contained; no claim of exhaustive duplicate detection.'))
    parent=read(source['parent_manifest'])
    exclusions=dict(derived=source['exclusions'],source_conflicting_hashes=parent['conflicting_label_hashes_excluded'],
        source_duplicates=parent['duplicates'],perceptual_exclusions=parent['perceptual_source_exclusions'])
    publish('vfd',rows,exclusions,[str(p),str(OUT/'source_review/vfd_source_group_evidence.json'),str(OUT/'source_review/vfd_cross_url_candidates.json')],
        'Source-URL connected-component holdout. The 1931-clip component remains intact, so 70/15/15 may not be attainable. Same-event ambiguities remain within components; sampled source checks cannot exclude undiscovered re-edits. No participant independence claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['vfd'],required=True);a=p.parse_args();offline();vfd()
