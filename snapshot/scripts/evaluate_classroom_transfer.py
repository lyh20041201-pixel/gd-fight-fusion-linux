"""Paired original vs lecture-only vs classroom-mix heads on existing GMD test.

Same source poses, windows, rising guard, threshold, cadence and backbone pass.
This is a known regression set (not new unseen research evidence).
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import copy
import json
import time
import numpy as np
import torch
from backend.vision.posec3d_actions import observed_windows,pose_heatmaps,FALL_INDEX,CLASSES,PoseC3D
from backend.vision.action_guards import rising_only_tracks,exclude_tracks
from scripts.fine_actions_bc import metrics
from scripts.train_classroom_posec3d import setup,BASE,OUT,load_posec3d
from scripts.prepare_classroom_posec3d import sha,write


def main():
    setup();model=load_posec3d(BASE)
    heads={'original':copy.deepcopy(model.cls_head.fc_cls)}
    files={'lecture_only':ROOT/'models/classroom_posec3d/le2i_head_pilot.pth',
           'classroom_mix':ROOT/'models/classroom_posec3d/classroom_mix_head_pilot.pth'}
    hashes={'original':sha(BASE)}
    for name,path in files.items():
        state=torch.load(path,map_location='cpu',weights_only=True)['state_dict']
        assert all(torch.equal(t.cpu(),state[k]) for k,t in model.state_dict().items() if k.startswith('backbone.'))
        head=copy.deepcopy(model.cls_head.fc_cls)
        head.load_state_dict({k.removeprefix('cls_head.fc_cls.'):v for k,v in state.items() if k.startswith('cls_head.fc_cls.')})
        heads[name]=head;hashes[name]=sha(path)
    rows=json.loads((ROOT/'results/live_actions/fine_labels_bc_v1/fine_annotations.json').read_text(encoding='utf-8'))['videos']
    rows=[r for r in rows if r['split']=='test']
    predictions={name:[] for name in heads}
    for row in rows:
        z=np.load(ROOT/f'datasets/video_events/fine_labels_bc_v1/pose/{row["id"]}.npz',allow_pickle=False)
        times=z['times']
        ends=sorted(set([*np.arange(4.,float(times[-1]),2.),float(times[-1])]))
        for end in ends:
            reasons={name:'warmup' if end<3.5 else 'pose_unknown_or_rising' for name in heads}
            results={name:[] for name in heads}
            if end>=3.5:
                idx=np.flatnonzero((times>=end-4-1e-8)&(times<=end+1e-8))
                clip=dict(keypoints=torch.from_numpy(z['keypoints'][:,idx]),boxes=torch.from_numpy(z['boxes'][:,idx]),
                    frame_indices=z['frame_indices'][idx],timestamps=times[idx]-times[idx][0],track_ids=z['track_ids'].tolist())
                clip=exclude_tracks(clip,rising_only_tracks(clip))
                for window in observed_windows(clip):
                    with torch.inference_mode(),torch.autocast('cuda'):
                        heat=torch.from_numpy(pose_heatmaps(window['points'],z['shape'])).cuda()
                        feats=model.backbone(heat).mean((2,3,4))
                        for name,head in heads.items():
                            p=head(feats).float().softmax(-1).mean(0).cpu().numpy()
                            assert np.isfinite(p).all()
                            results[name].append(dict(score=float(p[FALL_INDEX]),is_fall=int(p.argmax())==FALL_INDEX,
                                predicted_action=CLASSES[int(p.argmax())]))
            for name in heads:
                pred=dict(video_id=row['id'],end=end,reason=reasons[name],score=None)
                if results[name]:
                    best=max(results[name],key=lambda p:(p['is_fall'] and p['score']>=.3,p['score']))
                    pred.update(reason=None,raw=best,score=best['score'] if best['is_fall'] else 0.)
                predictions[name].append(pred)
        print(row['id'],flush=True)
    result=dict(protocol='Same 37 existing GMD test clips as original SAFER replay; original 8Hz pose; rising guard; 2-second cadence and endpoint; unchanged threshold 0.3 + argmax',
        caveat='Known regression set; not a new blind cross-scene benchmark; no live RGB quality guard',
        checkpoint_hashes=hashes,results={name:dict(summary=metrics(rows,preds,.3),predictions=preds) for name,preds in predictions.items()})
    write(OUT/'transfer_regression.json',result)
    for name,part in result['results'].items():
        print(name,json.dumps(part['summary']),flush=True)


if __name__=='__main__':
    main()
