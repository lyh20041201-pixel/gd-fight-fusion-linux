"""Read-only source/cache audit. Creates only round-two evidence; never extracts pose.

No split is inferred from filenames, archive numbers, model scores or pHashes.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.skeleton_common import CACHE, OUT, ROOT, POSE, PROTOCOL, digest, offline, read, sha, windows, frame_indices, write, seal

ROUND2 = ROOT / 'results/video_events/skeleton_comparison_round2'
MANIFESTS = CACHE / 'round2/manifests'


def youtube_id(url):
    parsed = urlparse(url.strip())
    host = (parsed.hostname or '').lower().removeprefix('www.').removeprefix('m.')
    parts = parsed.path.strip('/').split('/')
    if host == 'youtu.be':
        value = parts[0]
    elif host in ('youtube.com', 'youtube-nocookie.com'):
        value = parse_qs(parsed.query).get('v', [''])[0] if parts[0] == 'watch' else (parts[1] if len(parts) > 1 and parts[0] in ('shorts', 'embed', 'live') else '')
    else:
        raise ValueError(f'Unsupported source URL: {url}')
    if not re.fullmatch(r'[A-Za-z0-9_-]{11}', value):
        raise ValueError(f'Invalid YouTube source ID: {url}')
    return 'youtube:' + value  # case is significant


def file_record(path):
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha(path))


def snapshot():
    target = ROUND2 / 'audit/first_round_inventory.json'
    if target.exists():
        return read(target)
    files = sorted(p for p in OUT.rglob('*') if p.is_file())
    files += sorted((CACHE/'manifests').glob('*.json'))
    files += [ROOT / name for name in (
        'scripts/skeleton_common.py', 'scripts/train_skeleton_comparison.py',
        'scripts/evaluate_skeleton_comparison.py', 'scripts/run_skeleton_comparison.py',
        'scripts/extract_skeleton_comparison.py', 'backend/vision/skeleton_actions.py')]
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(file_record, files))
    value = dict(files=records, total_bytes=sum(r['bytes'] for r in records))
    seal(target, value)
    return value


def audit_one(args):
    import cv2
    import numpy as np
    import torch
    from backend.vision.skeleton_actions import prepare_skeleton
    name, row, pose_sha, extractor_hash = args
    target = ROUND2 / 'audit/samples' / name / (row['sample_id']+'.json')
    source = Path(row['path'])
    cache_path = CACHE / name / (row['sample_id']+'.pt')
    signature = digest(dict(row=row, pose=pose_sha, protocol=PROTOCOL, extractor=extractor_hash, audit_version=1))
    stat = source.stat() if source.is_file() else None
    cache_stat = cache_path.stat() if cache_path.is_file() else None
    freshness = dict(source_size=stat.st_size if stat else None, source_mtime_ns=stat.st_mtime_ns if stat else None,
                     cache_size=cache_stat.st_size if cache_stat else None, cache_mtime_ns=cache_stat.st_mtime_ns if cache_stat else None)
    if target.exists():
        old = read(target)
        if old['audit_signature'] != signature or old['freshness'] != freshness:
            raise ValueError(f'Audit input changed, use a new audit revision: {target}')
        return old
    problems = []
    actual_sha = sha(source) if stat else None
    if actual_sha != row['sha256']:
        problems.append('source_sha256_mismatch_or_missing')
    expected = digest(dict(sample=row['sample_id'], source=row['sha256'], protocol=PROTOCOL, pose=pose_sha, extractor_version=1))
    quality = []
    if cache_stat:
        cache = torch.load(cache_path, map_location='cpu', weights_only=True)
        for key, value in dict(signature=expected, source_sha256=row['sha256'], sample_id=row['sample_id'], pose_sha256=pose_sha, protocol=PROTOCOL, status='complete').items():
            if cache.get(key) != value:
                problems.append('cache_'+key+'_mismatch')
        spans = windows(row)
        if [(c['start'], c['end']) for c in cache['clips']] != spans:
            problems.append('window_mismatch')
        # n is not persisted by v1. The expected indices can still be checked from the
        # terminal frame: all non-tail windows have the exact same sampling formula.
        for clip in cache['clips']:
            ids = np.asarray(clip['frame_indices'])
            if len(ids) != 32 or clip['rgb'].shape != (32,112,112,3):
                problems.append('sample_shape_mismatch')
            if not np.allclose(np.asarray(clip['timestamps']), ids/cache['fps'], rtol=0, atol=1e-8):
                problems.append('timestamps_mismatch')
            if len(ids) == 32 and not np.array_equal(ids, frame_indices(clip['start'],clip['end'],cache['fps'],int(ids[-1])+1)):
                problems.append('frame_sampling_mismatch')
            pose = prepare_skeleton(clip)
            count = len(pose['valid'])
            pairs = 0
            if count >= 2:
                ij = torch.triu_indices(count,count,offset=1)
                pairs = int(((pose['valid'][ij[0]] & pose['valid'][ij[1]]).sum(-1)>=8).sum())
            reason = None
            if count == 0: reason = 'no_track_with_eight_unique_valid_frames'
            elif name == 'vfd' and count < 2: reason = 'fewer_than_two_usable_tracks'
            elif name == 'vfd' and pairs == 0: reason = 'insufficient_simultaneous_pair_observations'
            quality.append(dict(usable_tracks=count, eligible_pairs=pairs, unknown_reason=reason))
        # Existing RGB pixels only. This is a visual index, not a new sampling cache.
        first, last = cache['clips'][0], cache['clips'][-1]
        frames = [first['rgb'][0].numpy(), first['rgb'][16].numpy(), last['rgb'][-1].numpy()]
        thumb = ROUND2 / 'source_review/thumbs' / name / (row['sample_id']+'.jpg')
        thumb.parent.mkdir(parents=True, exist_ok=True)
        if not thumb.exists():
            if not cv2.imwrite(str(thumb), np.hstack(frames)[...,::-1]):
                raise IOError(str(thumb))
    else:
        problems.append('cache_missing')
    record = dict(dataset=name, sample_id=row['sample_id'], source_path=row['path'],
                  original_label=row['label'], source_label=row.get('source_label',row['label']),
                  original_group=row['group'], original_split=row['split'],
                  normalized_source_id=youtube_id(row['group']) if name=='vfd' else None,
                  proposed_source_group=None, split='unassigned',
                  source_sha256_declared=row['sha256'], source_sha256_observed=actual_sha,
                  cache_path=str(cache_path), expected_cache_signature=expected,
                  cache_signature_verified=not problems, problems=sorted(set(problems)),
                  skeleton_usable=any(q['unknown_reason'] is None for q in quality), window_quality=quality,
                  scope_basis=row['scope_basis'], origin=row['origin'],
                  label_review_status='source_label_with_inherited_AI_scope_screen; no_round2_individual_label_confirmation',
                  source_review_status='pending', human_confirmed=False,
                  exclusion_reason=None, audit_signature=signature, freshness=freshness)
    write(target,record)
    return record


def make_sheets(name, rows):
    import cv2
    import numpy as np
    groups=defaultdict(list)
    for row in rows:
        groups[row['group'] if name=='fallvision' else youtube_id(row['group'])].append(row)
    folder=ROUND2/'source_review'/name
    folder.mkdir(parents=True,exist_ok=True)
    index=[]
    # Distributed deterministic selection: evidence of within-group diversity, not
    # a claim of per-clip source verification. No scores are read for selection.
    for gi,(group,items) in enumerate(sorted(groups.items())):
        chosen=[items[i] for i in sorted(set(np.linspace(0,len(items)-1,min(12,len(items))).astype(int)))]
        canvas=np.full((len(chosen)*140,672,3),24,np.uint8)
        mapping=[]
        for i,row in enumerate(chosen):
            image=cv2.imread(str(ROUND2/'source_review/thumbs'/name/(row['sample_id']+'.jpg')))
            if image is not None:canvas[i*140+24:i*140+136]=np.hstack([image,image])[:,:672]
            # 3 observed times, displayed once at 2x horizontal pixels for readability.
            if image is not None:canvas[i*140+24:i*140+136]=cv2.resize(image,(672,112))
            label=f'{i+1} L{row["label"]} {Path(row["path"]).name} {row["sample_id"][:8]}'
            cv2.putText(canvas,label[:95],(4,i*140+17),0,.43,(255,255,255),1)
            mapping.append(dict(row=i+1,sample_id=row['sample_id'],path=row['path'],label=row['label']))
        target=folder/f'{gi:03d}.jpg'
        if not target.exists():cv2.imwrite(str(target),canvas)
        index.append(dict(group=group,rows=len(items),sheet=str(target),selected=mapping))
    write(folder/'index.json',index)


def vfd_candidates(rows):
    import numpy as np
    old_path=ROOT/'datasets/video_events/rebuild/tnue_vfd_fingerprints.json'
    old=read(old_path)
    row_by_path={r['path']:r for r in rows}
    fingerprints=[r for r in old['vfd'] if r['path'] in row_by_path]
    unique_paths={r['path'] for r in fingerprints}
    vals=np.array([r['hash'] for r in fingerprints],dtype=np.uint64)
    group_ids={g:i for i,g in enumerate(sorted({youtube_id(r['group']) for r in rows}))}
    gs=np.array([group_ids[youtube_id(row_by_path[r['path']]['group'])] for r in fingerprints])
    candidates={}
    started=time.monotonic()
    for i,row in enumerate(fingerprints):
        distances=np.bitwise_count(vals[i+1:] ^ vals[i])
        for relative in np.flatnonzero((distances<=12)&(gs[i+1:]!=gs[i])):
            j=i+1+int(relative);other=fingerprints[j]
            a=row_by_path[row['path']];b=row_by_path[other['path']]
            key=tuple(sorted((youtube_id(a['group']),youtube_id(b['group']))))
            d=int(distances[relative])
            if key not in candidates or d<candidates[key]['distance']:
                candidates[key]=dict(source_a=key[0],source_b=key[1],distance=d,
                    a=dict(**row,sample_id=a['sample_id']),b=dict(**other,sample_id=b['sample_id']),
                    review_status='pending_AI_visual_review',same_event=None,human_confirmed=False)
    value=dict(status='candidate_screen_only',source=str(old_path),source_sha256=sha(old_path),
               rows_with_fingerprints=len(unique_paths),expected_rows=len(rows),fingerprints=len(fingerprints),
               missing_paths=sorted(set(row_by_path)-unique_paths),
               method='Reuse all five-frame source pHashes; compare distinct canonical YouTube IDs; 64-bit Hamming distance <=12',
               limitation='Similarity is not duplicate proof; sampled hashes can miss transformations and compilations; unresolved candidates block sealing.',
               seconds=time.monotonic()-started,candidates=sorted(candidates.values(),key=lambda x:(x['distance'],x['source_a'],x['source_b'])))
    write(ROUND2/'source_review/vfd_cross_url_candidates.json',value)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    offline()
    if (ROUND2/'source_gate.json').exists():
        raise ValueError('Source-review decisions already recorded. Refusing to overwrite reviewed drafts; use a new audit revision for changed inputs.')
    import torch,cv2
    torch.set_num_threads(1);cv2.setNumThreads(1)
    ROUND2.mkdir(parents=True,exist_ok=True);MANIFESTS.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    print('STAGE immutable_inventory model=none seed=none epoch=0',flush=True)
    inventory=snapshot()
    pose_sha=sha(POSE)
    environment=read(OUT/'environment.json')
    extractor=sha(ROOT/'scripts/extract_skeleton_comparison.py')
    if pose_sha!=read(OUT/'pose_provenance.json')['sha256']:
        raise ValueError('Pose hash differs from first round')
    final_verification=read(OUT/'final_verification.json')
    recorded_extractor=final_verification['artifact_code_hashes']['scripts\\extract_skeleton_comparison.py']
    if final_verification['status']!='passed' or extractor!=recorded_extractor:
        raise ValueError('Extractor differs from final first-round recorded version; provenance review required')
    write(ROUND2/'audit/extractor_provenance.json',dict(current_sha256=extractor,
        final_verification_sha256=sha(OUT/'final_verification.json'),
        recorded_final_sha256=recorded_extractor,earlier_environment_sha256=environment['entrypoints']['extract_skeleton_comparison.py'],
        recovery_record=read(OUT/'file_lock_recovery.json'),
        explanation='Final first-round verification records the post-recovery status-writer import; earlier environment snapshot predates this change.',
        extractor_version=1,weights_sha256=pose_sha))
    manifests={name:read(CACHE/'manifests'/f'{name}.json') for name in ('fallvision','vfd')}
    summary={}
    for name,manifest in manifests.items():
        rows=manifest['rows'];records=[];stage_started=time.monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i,record in enumerate(pool.map(audit_one,[(name,r,pose_sha,extractor) for r in rows])):
                records.append(record)
                if (i+1)%100==0 or i+1==len(rows):
                    elapsed=time.monotonic()-stage_started
                    state=dict(stage='source_hash_and_existing_cache_audit',dataset=name,completed=i+1,total=len(rows),
                        model=None,seed=None,epoch=0,elapsed_seconds=elapsed,segments_per_second=(i+1)/elapsed,
                        estimated_remaining_seconds=(len(rows)-i-1)*elapsed/(i+1))
                    write(ROUND2/'progress.json',state)
                    print(json.dumps(state),flush=True)
        summary[name]=dict(rows=len(rows),labels=dict(Counter(r['label'] for r in rows)),
            original_groups=len({r['group'] for r in rows}),
            canonical_groups=len({r['normalized_source_id'] for r in records}) if name=='vfd' else None,
            verified_sources=sum(r['source_sha256_observed']==r['source_sha256_declared'] for r in records),
            compatible_caches=sum(r['cache_signature_verified'] for r in records),
            unusable_skeleton=dict(Counter(r['original_label'] for r in records if not r['skeleton_usable'])),
            problems=[dict(sample_id=r['sample_id'],problems=r['problems']) for r in records if r['problems']],
            inherited_exclusions=len(manifest['exclusions']))
        write(MANIFESTS/f'{name}_source_review_draft.json',dict(status='NOT_SEALED_NOT_TRAINING_READY',
            dataset=manifest['dataset'],labels=manifest['labels'],parent_manifest=str(CACHE/'manifests'/f'{name}.json'),
            parent_manifest_sha256=sha(CACHE/'manifests'/f'{name}.json'),split_seed=42,
            target_ratios=dict(train=.70,validation=.15,test=.15),
            test_description='Repartitioned retained test; dataset evaluated in first round; not a new blind external test',
            rows=[dict(**r,original_label=r['label'],original_group=r['group'],original_split=r['split'],
                proposed_source_group=rec['normalized_source_id'],new_split='unassigned',
                source_review_status='pending',human_confirmed=False,audit_record=rec) for r,rec in zip(rows,records)],
            inherited_exclusions=manifest['exclusions'],
            parent_source_manifest=read(manifest['parent_manifest']),
            training_allowed=False))
        make_sheets(name,rows)
    vfd_candidates(manifests['vfd']['rows'])
    history={}
    for name in ('gmd','tnue'):
        result=read(OUT/'evaluations'/name/'summary.json')
        history[name]=dict(samples=result['samples'],aggregate=result['aggregate'],
            per_seed=result['per_seed'],source=str(OUT/'evaluations'/name/'summary.json'),
            comparison_warning='Entire historical external population; cannot be compared directly with a future round2 test subset')
    write(ROUND2/'audit/summary.json',dict(status='source_bytes_and_cache_audited_group_review_pending',
        datasets=summary,pose_sha256=pose_sha,extractor_sha256=extractor,
        protocol=PROTOCOL,first_round_inventory_files=len(inventory['files']),history=history,
        elapsed_seconds=time.monotonic()-started,network='socket connections disabled; no downloads',
        fresh_pose_extractions=0,fresh_video_sampling_caches=0,trained_models=0))
    write(ROUND2/'progress.json',dict(stage='source_group_review_pending',completed=7029,total=7029,model=None,seed=None,epoch=0,training_started=False))
    print('AUDIT COMPLETE; SOURCE GROUP REVIEW REQUIRED BEFORE ANY SPLIT OR TRAINING',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
