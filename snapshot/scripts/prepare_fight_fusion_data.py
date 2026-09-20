"""Seal source-grouped fusion data without running an action or pose model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.download_fight_fusion_data import ROOT, DATA, OUT, sha256, write_json, check_disk

VFD = ROOT / 'datasets/video_events/skeleton_rebuild/round2/manifests/vfd.json'
AIRTLAB = ROOT / 'datasets/video_events/airtlab.json'
SPLITS = ('train', 'branch_val', 'fusion_fit', 'calibration', 'test', 'legacy_test')
PROTOCOL = {'version': 1, 'seed': 42, 'phash_bits': 64, 'max_hamming': 4,
            'max_dhash_hamming': 8, 'min_matched_frames': 3,
            'require_distinct_temporal_fingerprints': 3, 'fingerprint_bucket_bits': [13,13,13,13,12],
            'fingerprint_step_seconds': .5, 'max_fingerprints_per_video': 256,
            'new_non_test_weights': {'train': 50, 'branch_val': 10, 'fusion_fit': 20, 'calibration': 10}}
INSPECTION_PROTOCOL = {'version': 1, 'fingerprint_step_seconds': .5, 'max_fingerprints_per_video': 256,
                       'representation': 'uniform-margin-trimmed 64-bit pHash and 64-bit dHash'}


class LabelConflict(ValueError):
    def __init__(self, evidence):
        self.evidence=evidence
        super().__init__(f"Directory/frame label conflict: {evidence['path']}")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_id(value):
    if not value:
        return None
    if value.startswith('youtube:'):
        return value
    p = urlparse(value.strip())
    host = (p.hostname or '').lower().removeprefix('www.').removeprefix('m.')
    parts = p.path.strip('/').split('/')
    if host == 'youtu.be':
        key = parts[0]
    elif host in ('youtube.com', 'youtube-nocookie.com'):
        key = parse_qs(p.query).get('v', [''])[0] if parts[0] == 'watch' else (parts[1] if len(parts)>1 else '')
    else:
        return value
    return 'youtube:' + key if re.fullmatch(r'[A-Za-z0-9_-]{11}', key) else value


def parse_scfd_sources(path):
    mapping = defaultdict(set)
    current = None
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if line.startswith(('https://', 'http://')):
            current = source_id(line)
        match = re.match(r'((?:no)?fi\d+)\s*:', line, re.I)
        if match and current:
            key = match.group(1).lower()
            mapping[key].add(current)
        interval = re.match(r'nofi(\d+)\s*-\s*nofi(\d+)\s*:\s*(.+)', line, re.I)
        if interval:
            first,last,dataset=interval.groups()
            for number in range(int(first),int(last)+1):
                mapping[f'nofi{number:03d}'].add('scfd-author-collection:'+dataset.strip().lower())
    return {k:sorted(v) for k,v in mapping.items()}


def intervals_from_labels(labels, fps):
    result = []
    start = None
    for i, val in enumerate(list(labels) + [0]):
        if val not in (0, 1):
            raise ValueError('Frame annotation must contain only 0 and 1')
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            result.append([start / fps, i / fps])
            start = None
    return result


def initial_rows(allow_partial=False):
    rows, receipts, missing = [], [], []
    for dataset, path in [('vfd', VFD), ('airtlab', AIRTLAB)]:
        m = read(path)
        if dataset == 'vfd' and (m.get('status') != 'sealed' or not m.get('training_allowed')):
            raise ValueError('Original VFD manifest is not sealed for training')
        receipts.append({'dataset': dataset, 'manifest': str(path), 'manifest_sha256': sha256(path),
                         'reuse': True, 'prior_training_history': True})
        for original in m['rows']:
            r = dict(sample_id=(original.get('sample_id') if dataset == 'vfd' else digest([dataset, original['path']])[:24]),
                     dataset=dataset, path=original['path'], sha256=original['sha256'],
                     group=(original['group'] if dataset == 'vfd' else 'airtlab:' + original['group']),
                     split={'validation': 'branch_val', 'test': 'legacy_test'}.get(original['split'], original['split']),
                     original_split=original['split'], label=int(original['label']), label_kind='video',
                     source_id=source_id(original.get('normalized_source_id') or original.get('original_group')),
                     frozen_legacy=True)
            rows.append(r)
    for dataset in ['scfd', 'ubi']:
        path = OUT / f'{dataset}_receipt.json'
        if not path.exists() or read(path).get('status') != 'complete':
            missing.append(dataset)
            continue
        receipt = read(path)
        receipts.append({'dataset': dataset, 'receipt': str(path), 'receipt_sha256': sha256(path),
                         'archive_sha256': receipt['archive_sha256'], 'url': receipt['url']})
        files = {Path(x['path']).name: x for x in receipt['files']}
        videos = [x for x in receipt['files'] if x['path'].lower().endswith('.mp4')]
        source_map = parse_scfd_sources(files['videos.txt']['path']) if dataset == 'scfd' else {}
        tests = set(Path(files['test_videos.csv']['path']).read_text(encoding='utf-8-sig').split()) if dataset == 'ubi' else set()
        if dataset == 'ubi' and len(tests) != 67:
            raise ValueError('UBI official test list must contain 67 video stems')
        for item in videos:
            video = Path(item['path'])
            sources = source_map.get(video.stem.lower(), [])
            source = sources[0] if len(sources)==1 else None
            r = dict(sample_id=digest([dataset, item['sha256'], str(video.relative_to(DATA))])[:24], dataset=dataset,
                     path=str(video), sha256=item['sha256'], group=source or dataset+':unresolved:'+video.stem,
                     source_id=source, source_ids=sources, split='unassigned', original_split=('test' if video.stem in tests else 'unspecified'),
                     label=int(video.parent.name in ('fight',)), label_kind='frame' if dataset == 'ubi' else 'video',
                     frozen_legacy=False, source_resolution=('author_source_or_collection' if len(sources)==1 else
                         'ambiguous_author_sources_conservatively_merged' if sources else 'video_only_unknown_original_source'))
            if dataset == 'ubi':
                r['annotation_path'] = files[video.stem+'.csv']['path']
                r['annotation_sha256'] = files[video.stem+'.csv']['sha256']
                # Preserve the unexplained extra filename flag; never guess a field's meaning.
                r['source_filename_flags'] = video.stem.split('_')[2:]
            rows.append(r)
    if missing and not allow_partial:
        raise RuntimeError(f'Downloads incomplete: {missing}; --allow-partial is smoke-only')
    return rows, receipts, missing


def frame_fingerprint(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ys, xs = np.flatnonzero(gray.std(1)>5), np.flatnonzero(gray.std(0)>5)
    if len(ys)<16 or len(xs)<16 or gray.std()<12:
        return None
    gray = gray[ys[0]:ys[-1]+1, xs[0]:xs[-1]+1]
    small = cv2.resize(gray, (32, 32)).astype(np.float32)
    dct = cv2.dct(small)[:8, :8].flatten()
    bits = dct > np.median(dct[1:]); bits[0] = False
    ph = int.from_bytes(np.packbits(bits).tobytes(), 'little')
    dh = cv2.resize(gray, (9, 8)).astype(np.int16)
    dh = int.from_bytes(np.packbits(dh[:,1:] > dh[:,:-1]).tobytes(), 'little')
    return ph, dh


def sampled_frames(cap, times, fps, frame_count):
    """Decode nearby positions once rather than repeatedly replaying the same GOP."""
    current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    for t in times:
        target = max(0, min(frame_count-1, int(float(t)*fps+.5)))
        if target < current or target-current > 60:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        else:
            for _ in range(target-current):
                if not cap.grab():
                    break
        ok, frame = cap.read()
        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        yield float(t), ok, frame


def inspect_video(row):
    path = Path(row['path'])
    cache_path = OUT / 'inspection' / (row['sample_id']+'.json')
    stat = path.stat()
    signature = digest([row['sha256'], stat.st_size, stat.st_mtime_ns, INSPECTION_PROTOCOL])
    # The first extraction pass used the same pixel sampling with the split protocol
    # in its cache key. Reuse those identical fingerprints, not any other revision.
    legacy_protocol = {k:v for k,v in PROTOCOL.items() if k not in ('require_distinct_temporal_fingerprints','fingerprint_bucket_bits')}
    compatible_signature = digest([row['sha256'], stat.st_size, stat.st_mtime_ns, legacy_protocol])
    if cache_path.exists():
        old = read(cache_path)
        if old.get('signature') in (signature, compatible_signature):
            if row.get('annotation_path') and sha256(Path(row['annotation_path'])) != row['annotation_sha256']:
                raise ValueError('UBI annotation SHA mismatch')
            return old
    actual = sha256(path)
    if actual != row['sha256']:
        raise ValueError(f'Source SHA mismatch: {path}')
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)); frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if not math.isfinite(fps) or fps<=0 or frames<1 or min(h,w)<1:
        cap.release()
        raise ValueError(f'Invalid video metadata: {path}')
    duration = frames/fps
    labels=None
    if row['label_kind']=='frame':
        if sha256(Path(row['annotation_path'])) != row['annotation_sha256']:
            cap.release();raise ValueError('UBI annotation SHA mismatch')
        labels=[int(v.strip()) for v in Path(row['annotation_path']).read_text(encoding='utf-8-sig').splitlines() if v.strip()]
        if len(labels)!=frames:
            cap.release();raise ValueError(f'Frame annotation/video length mismatch {path.name}: {len(labels)} vs {frames}')
        if any(v not in (0,1) for v in labels):
            cap.release();raise ValueError(f'Nonbinary frame annotation: {path.name}')
        if bool(row['label']) != bool(any(labels)):
            cap.release()
            raise LabelConflict(dict(sample_id=row['sample_id'],dataset=row['dataset'],path=str(path),
                source_sha256=actual,annotation_path=row['annotation_path'],annotation_sha256=row['annotation_sha256'],
                original_video_label=row['label'],positive_frame_count=sum(labels),annotation_frame_count=len(labels),
                frame_count=frames,fps=fps,duration=duration,original_split=row['original_split'],
                reason='directory_label_conflicts_with_frame_labels',
                disposition='quarantine_without_relabeling; excluded from training and evaluation'))
    step = PROTOCOL['fingerprint_step_seconds']
    times = np.arange(0, max(0,duration-1/fps)+step/4, step)
    if len(times)>PROTOCOL['max_fingerprints_per_video']:
        times = np.linspace(0, max(0,duration-1/fps), PROTOCOL['max_fingerprints_per_video'])
    if len(times)<5:
        times = np.linspace(0, max(0,duration-1/fps), min(5,frames))
    fingerprints = []
    read_failures = 0
    for t,ok,frame in sampled_frames(cap,times,fps,frames):
        if not ok:
            read_failures += 1
            continue
        result = frame_fingerprint(frame)
        if result is not None:
            fingerprints.append([float(t), result[0], result[1]])
    cap.release()
    if read_failures == len(times):
        raise ValueError(f'Video cannot decode: {path}')
    out = dict(signature=signature, source_sha256=actual, duration=duration, fps=fps, frame_count=frames,
               source_shape=[h,w], fingerprints=fingerprints, fingerprint_read_failures=read_failures)
    if labels is not None:
        out['positive_intervals'] = intervals_from_labels(labels, fps)
    write_json(cache_path, out)
    return out


class DSU:
    def __init__(self, n): self.parent = list(range(n))
    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]; a = self.parent[a]
        return a
    def union(self, a, b):
        a,b = self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)] = min(a,b)


def screen_overlap(rows, inspections):
    dsu = DSU(len(rows)); evidence = []; exact = {}; groups = {}; source = {}
    for i,r in enumerate(rows):
        identities=[('sha256',r['sha256'],exact), ('group',r['group'],groups)]
        identities += [('source_id',s,source) for s in r.get('source_ids',[]) or [r.get('source_id')]]
        for kind, value, index in identities:
            if not value:continue
            if value in index:
                j=index[value];dsu.union(i,j)
                if not (r['frozen_legacy'] and rows[j]['frozen_legacy']):
                    evidence.append({'a':rows[j]['sample_id'],'b':r['sample_id'],'method':kind,'value':value})
            else:index[value]=i
    buckets = defaultdict(list); matches = defaultdict(list)
    # Legacy samples enter the index first; no legacy-vs-legacy regrouping by appearance.
    for i,r in enumerate(rows):
        for fi,(t,ph,dh) in enumerate(inspections[r['sample_id']]['fingerprints']):
            candidates=set()
            for k in range(5): candidates.update(buckets[(k,(ph>>(13*k))&8191)])
            if not r['frozen_legacy']:
                for j,fj,ot,op,od in candidates:
                    if i==j or dsu.find(i)==dsu.find(j):continue
                    if (ph^op).bit_count()<=4 and (dh^od).bit_count()<=8:
                        matches[(j,i)].append((fj,fi,ot,t))
            for k in range(5):buckets[(k,(ph>>(13*k))&8191)].append((i,fi,t,ph,dh))
    for (a,b), observed in matches.items():
        # Require coherent temporal offsets; a single similar background frame is insufficient.
        offsets=defaultdict(list)
        for fa,fb,ta,tb in observed:offsets[round((tb-ta)/.5)].append((fa,fb,ta,tb))
        coherent=[]
        for key, group in offsets.items():
            candidate=group+offsets.get(key+1,[])
            if len({x[0] for x in candidate})>=3 and len({x[1] for x in candidate})>=3:
                fingerprints_a=inspections[rows[a]['sample_id']]['fingerprints']
                fingerprints_b=inspections[rows[b]['sample_id']]['fingerprints']
                if (len({tuple(fingerprints_a[x[0]][1:]) for x in candidate})<3 or
                        len({tuple(fingerprints_b[x[1]][1:]) for x in candidate})<3):
                    continue
                if min(max(x[2] for x in candidate)-min(x[2] for x in candidate),
                       max(x[3] for x in candidate)-min(x[3] for x in candidate))>=.5:
                    coherent=candidate;break
        if coherent:
            dsu.union(a,b)
            evidence.append({'a':rows[a]['sample_id'],'b':rows[b]['sample_id'],'method':'perceptual_temporal_candidate',
                             'matched_frames':len(coherent),'action':'conservative_source_group_or_exclusion'})
    return dsu,evidence


def allocate_groups(groups, weights, seed=42):
    """Greedy class-balanced whole-group allocation, deterministic independent of row order."""
    splits=list(weights);total=sum(weights.values())
    sizes={g:Counter(r['label'] for r in rs) for g,rs in groups.items()}
    counts=Counter(r['label'] for rs in groups.values() for r in rs)
    targets={s:{c:counts[c]*weights[s]/total for c in [0,1]} for s in splits}
    used={s:Counter() for s in splits}; assigned={}
    order=sorted(groups,key=lambda g:(-len(groups[g]),digest([seed,g])))
    for g in order:
        def cost(s):
            return sum(((used[s][c]+sizes[g][c]-targets[s][c])**2-(used[s][c]-targets[s][c])**2)/max(1,targets[s][c]) for c in [0,1])
        split=min(splits,key=lambda s:(cost(s),splits.index(s)))
        used[split].update(sizes[g]);assigned[g]=split
    return assigned


def deduplicate_and_split(rows, dsu):
    components=defaultdict(list)
    for i,r in enumerate(rows):components[dsu.find(i)].append(r)
    kept=[];excluded=[];inherited_conflicts=[];new_groups={}
    for members in components.values():
        old=[r for r in members if r['frozen_legacy']];new=[r for r in members if not r['frozen_legacy']]
        kept.extend(old)
        if old:
            if len({r['split'] for r in old})>1:
                inherited_conflicts.append([r['sample_id'] for r in old])
            for r in new:excluded.append({'sample_id':r['sample_id'],'dataset':r['dataset'],'path':r['path'],
                                         'reason':'overlap_with_frozen_legacy_source','matched_legacy':[x['sample_id'] for x in old]})
            continue
        unique=[];by_hash={}
        labels_by_hash=defaultdict(set)
        for r in new:labels_by_hash[r['sha256']].add(r['label'])
        official_test=any(r['dataset']=='ubi' and r['original_split']=='test' for r in new)
        for r in sorted(new,key=lambda x:x['sample_id']):
            if len(labels_by_hash[r['sha256']])>1:
                excluded.append({'sample_id':r['sample_id'],'dataset':r['dataset'],'path':r['path'],
                                 'reason':'conflicting_new_labels_for_identical_bytes'})
                continue
            if r['sha256'] in by_hash:
                excluded.append({'sample_id':r['sample_id'],'dataset':r['dataset'],'path':r['path'],
                                 'reason':'duplicate_new_sha256','retained':by_hash[r['sha256']]['sample_id']})
            else:unique.append(r);by_hash[r['sha256']]=r
        if unique:
            name='fusion-source:'+digest(sorted(r['group'] for r in unique))[:24]
            for r in unique:
                r['group']=name
                r['official_test_source_group']=official_test
            new_groups[name]=unique
    fixed_test={g for g,rs in new_groups.items() if any(r.get('official_test_source_group') for r in rs)}
    # SCFD holds out 10% at source-group level; a group can never straddle datasets/splits.
    scfd={g:rs for g,rs in new_groups.items() if g not in fixed_test and any(r['dataset']=='scfd' for r in rs)}
    scfd_choice=allocate_groups(scfd,{'pool':90,'test':10})
    fixed_test.update(g for g,s in scfd_choice.items() if s=='test')
    pool={g:rs for g,rs in new_groups.items() if g not in fixed_test}
    choices=allocate_groups(pool,PROTOCOL['new_non_test_weights'])
    for g,rs in new_groups.items():
        for r in rs:r['split']='test' if g in fixed_test else choices[g]
        kept.extend(rs)
    return sorted(kept,key=lambda r:(r['dataset'],r['sample_id'])),excluded,inherited_conflicts


def validate_manifest(manifest):
    rows=manifest['rows']
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Duplicate sample IDs')
    for r in rows:
        if r['split'] not in SPLITS:raise ValueError('Unassigned sample')
        if r['label_kind']=='frame' and 'positive_intervals' not in r:raise ValueError('Missing temporal labels')
    # Do not silently rewrite historical leakage; any newly introduced cross-split source is forbidden.
    for key in ['group','sha256']:
        groups=defaultdict(list)
        for r in rows:groups[r[key]].append(r)
        for rs in groups.values():
            if len({r['split'] for r in rs})>1 and any(not r['frozen_legacy'] for r in rs):
                raise ValueError(f'New cross-split {key} leakage')
    vfd=Counter(r['split'] for r in rows if r['dataset']=='vfd')
    if vfd != Counter({'train':1931,'branch_val':207,'legacy_test':205}):raise ValueError('Frozen VFD split changed')
    if manifest['status']=='sealed':
        for s in ('fusion_fit','calibration','test'):
            if {r['label'] for r in rows if r['split']==s}!={0,1}:raise ValueError(f'{s} lacks both classes')


def main():
    p=argparse.ArgumentParser();p.add_argument('--allow-partial',action='store_true');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--wait-downloads',action='store_true',help='Wait for both verified download receipts before the complete preparation')
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);check_disk(OUT)
    cv2.setNumThreads(1)
    if (OUT/'manifest.seal.json').exists():
        seal=read(OUT/'manifest.seal.json')
        if sha256(OUT/'manifest.json')!=seal['manifest_sha256']:raise RuntimeError('Sealed manifest changed')
        print('Already sealed; no changes',flush=True);return
    def progress(stage,**kw):write_json(OUT/'prepare_progress.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**kw))
    if a.wait_downloads:
        while True:
            pending=[d for d in ('scfd','ubi') if not (OUT/f'{d}_receipt.json').exists() or read(OUT/f'{d}_receipt.json').get('status')!='complete']
            if not pending:break
            for d in pending:
                state=OUT/f'download_{d}.json'
                if state.exists() and read(state).get('status')=='failed':
                    raise RuntimeError(f'Download failed: {d}; see {state}')
            progress('waiting_for_verified_downloads',pending=pending)
            print('waiting for verified downloads',pending,flush=True)
            time.sleep(30)
    rows,receipts,missing=initial_rows(a.allow_partial);inspections={};errors=[];quarantine=[]
    original_counts=Counter(r['dataset'] for r in rows)
    progress('inspect',completed=0,total=len(rows),missing_datasets=missing)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(inspect_video,r):r for r in rows}
        for future in as_completed(futures):
            r=futures[future]
            try:
                ins=future.result();inspections[r['sample_id']]=ins
                for k in ('duration','fps','frame_count','source_shape','positive_intervals'):
                    if k in ins:r[k]=ins[k]
            except LabelConflict as exc:
                quarantine.append(exc.evidence)
                write_json(OUT/'uncertain_labels.json',{'original_video_counts':dict(original_counts),'quarantine':quarantine})
                print('quarantine',json.dumps(exc.evidence),flush=True)
            except Exception as exc:
                errors.append({'sample_id':r['sample_id'],'path':r['path'],'error':repr(exc)})
                write_json(OUT/'inspection_errors.json',{'errors':errors})
                print('inspection_error',json.dumps(errors[-1]),flush=True)
            n=len(inspections)+len(errors)+len(quarantine)
            if n%25==0 or n==len(rows):
                progress('inspect',completed=n,total=len(rows),errors=len(errors),quarantine=len(quarantine),missing_datasets=missing)
                print('inspect',n,'/',len(rows),'errors',len(errors),'quarantine',len(quarantine),flush=True)
    write_json(OUT/'inspection_errors.json',{'errors':errors})
    if errors:raise RuntimeError(f'{len(errors)} inspection errors; refusing to silently discard samples')
    write_json(OUT/'uncertain_labels.json',{'original_video_counts':dict(original_counts),'quarantine':quarantine})
    rows=[r for r in rows if r['sample_id'] in inspections]
    progress('source_and_perceptual_overlap',total=len(rows))
    dsu,evidence=screen_overlap(rows,inspections)
    rows,excluded,old_conflicts=deduplicate_and_split(rows,dsu)
    write_json(OUT/'overlap_audit.json',{'protocol':PROTOCOL,'evidence':evidence,'excluded':excluded,'inherited_legacy_conflicts':old_conflicts,
                                      'limitation':'Sampled pHash+dHash temporal screening is conservative, not proof of no transformed/re-edited duplicates.'})
    counts={d:{s:dict(Counter(r['label'] for r in rows if r['dataset']==d and r['split']==s)) for s in SPLITS}
            for d in sorted({r['dataset'] for r in rows})}
    manifest=dict(version=1,status='partial' if missing else 'sealed',training_allowed=not missing,
                  rows=rows,counts=counts,source_receipts=receipts,protocol=PROTOCOL,
                  source_receipt_fingerprint=digest(receipts),
                  quarantine=quarantine,
                  accounting={d:{'downloaded_or_reused_videos':original_counts[d],
                     'label_conflicts':sum(r['dataset']==d for r in quarantine),
                     'usable_before_overlap':original_counts[d]-sum(r['dataset']==d for r in quarantine),
                     'overlap_exclusions':sum(r['dataset']==d for r in excluded),
                     'final_videos':sum(r['dataset']==d for r in rows)} for d in original_counts},
                  limitations=['VFD and AIRTLab legacy tests were used by earlier experiments and are not blind.',
                               'Scene/person-disjoint independence is not established for unresolved original sources.',
                               'Perceptual fingerprints cannot rule out all transformed, cropped or re-edited copies.',
                               'Source labels are inherited; UBI frame boundaries are not individually human re-annotated.',
                               'Public datasets do not substitute for continuous classroom-camera acceptance footage.',
                               'UBI filename extra attribute is preserved without guessing its semantic meaning.'] +
                               ([f'Incomplete downloads: {missing}; smoke only, formal training prohibited.'] if missing else []))
    validate_manifest(manifest)
    write_json(OUT/'manifest.json',manifest)
    if not missing:
        write_json(OUT/'manifest.seal.json',{'version':1,'manifest_sha256':sha256(OUT/'manifest.json'),
                                           'rows':len(rows),'sealed_at':time.time(),'protocol_sha256':digest(PROTOCOL)})
    progress('partial_smoke_only' if missing else 'sealed',rows=len(rows),excluded=len(excluded),counts=counts)
    print(json.dumps({'status':manifest['status'],'rows':len(rows),'excluded':len(excluded),'counts':counts}),flush=True)


if __name__=='__main__':main()
