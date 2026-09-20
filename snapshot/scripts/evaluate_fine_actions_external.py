"""Retained FallVision clip-level regression AFTER all six GMD selections.

Uses the already audited cached windows, and does not tune any model/threshold.
These are coarse source video labels, not fine action/onset labels or hours of
continuous surveillance. The data has prior test history.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import Counter
import numpy as np
import torch
from scripts.fine_actions_bc import *


def prepare():
    audit=read(ROOT/'results/video_events/skeleton_stgcnpp_ab/audit/inputs_fallvision.json')
    manifest=read(ROOT/'datasets/video_events/skeleton_rebuild/round2/manifests/fallvision.json')
    rows=[r for r in manifest['rows'] if r['split']=='test'];lookup={r['sample_id']:r for r in audit['rows']}
    x=[];valid=[];windows=[];records=[]
    for r in rows:
        info=lookup[r['sample_id']]
        assert info['source_sha256']==r['sha256'] and sha(r['path'])==r['sha256']
        assert sha(info['prepared_cache'])==info['prepared_cache_sha256']
        data=torch.load(info['prepared_cache'],map_location='cpu',weights_only=True)
        for wi,(item,w) in enumerate(zip(data['items'],data['windows'])):
            record=dict(video_id=r['sample_id'],end=w['end'],start=w['start'],feature_index=-1,
                        label=-1,reason='pose_unknown',split='external_test')
            if len(item['valid']):
                # Prepared cache already applies >=8 unique frame eligibility.
                counts=item['valid'].sum(-1)
                primary=max(range(len(counts)),key=lambda i:(int(counts[i]),float(item['features'][i,2].mean())))
                record.update(feature_index=len(x),reason=None,primary_track_index=primary)
                x.append(item['features'][primary]);valid.append(item['valid'][primary])
            windows.append(record)
        records.append(dict(id=r['sample_id'],label=r['label'],path=r['path'],sha256=r['sha256'],
                            source_group=r['group'],source_label=r.get('source_label'),duration=r['duration']))
    np.savez_compressed(CACHE/'external_fallvision.npz',features=torch.stack(x).numpy(),valid=torch.stack(valid).numpy())
    seal(CACHE/'external_fallvision.json',dict(videos=records,windows=windows,
        protocol='Existing audited 4s / 32-frame coarse windows; short original clips retained as in historical test. Clip maximum; no new threshold tuning.',
        origin_manifest_sha256=sha(ROOT/'datasets/video_events/skeleton_rebuild/round2/manifests/fallvision.json'),
        artifact_sha256=sha(CACHE/'external_fallvision.npz')))
    print('EXTERNAL_PREPARED',len(records),len(windows),flush=True)


def evaluate():
    setup();verify_seal()
    selections=read(OUT/'all_selections_before_test.json');assert len(selections)==6
    meta=read(CACHE/'external_fallvision.json');z=np.load(CACHE/'external_fallvision.npz')
    assert sha(CACHE/'external_fallvision.npz')==meta['artifact_sha256']
    data=dict(windows=meta['windows'],x=torch.from_numpy(z['features']),valid=torch.from_numpy(z['valid']))
    summary={}
    for arm in ['A','B','C']:
        for seed in ([None] if arm=='A' else POLICY['seeds']):
            if arm=='A':model,spec=make_baseline();threshold=spec['threshold'];name='A'
            else:
                name=f'{arm}_{seed}';path=OUT/arm/f'seed_{seed}'/'best.pt'
                assert sha(path)==selections[name]['checkpoint_sha256']
                saved=torch.load(path,weights_only=True,map_location='cpu')
                model=FineActionModel(2 if arm=='B' else 8);model.load_state_dict(saved['state_dict']);threshold=saved['threshold']
            predictions=predict(model.cuda(),data);byid=defaultdict(list)
            for w in predictions:byid[w['video_id']].append(w)
            counts=dict(tp=0,fn=0,fp=0,tn=0,unknown=0);records=[]
            for r in meta['videos']:
                score=max((w['score'] for w in byid[r['id']] if w['score'] is not None),default=None)
                positive=score is not None and score>=threshold;truth=bool(r['label'])
                counts['tp' if positive and truth else 'fn' if truth else 'fp' if positive else 'tn']+=1
                counts['unknown']+=int(score is None)
                records.append(dict(**r,score=score,predicted_fall=positive))
            counts.update(recall=counts['tp']/(counts['tp']+counts['fn']),fpr=counts['fp']/(counts['fp']+counts['tn']),threshold=threshold)
            summary[name]=counts
            write(OUT/'evaluation'/f'{name}_fallvision_test.json',dict(metrics=counts,rows=records))
            print('EXTERNAL_TEST',name,counts,flush=True);del model
    write(OUT/'external_fallvision_summary.json',summary)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','evaluate']);args=parser.parse_args()
    prepare() if args.command=='prepare' else evaluate()
