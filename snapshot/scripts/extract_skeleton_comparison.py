"""Resume per source segment; RGB and pose use exactly the same decoded timestamps."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,time,traceback
import cv2,numpy as np,torch
from scripts.skeleton_common import *
from scripts.skeleton_io import write
from backend.vision.detector import Detection
from backend.vision.tracker import ByteTracker

def cache_signature(row,pose_sha):
    return digest(dict(sample=row['sample_id'],source=row['sha256'],protocol=PROTOCOL,pose=pose_sha,extractor_version=1))

def resize_rgb(im):
    h,w=im.shape[:2];s=112/max(h,w)
    small=cv2.resize(im,(max(1,round(w*s)),max(1,round(h*s))))
    canvas=np.zeros((112,112,3),np.uint8); y=(112-small.shape[0])//2;x=(112-small.shape[1])//2
    canvas[y:y+small.shape[0],x:x+small.shape[1]]=small[...,::-1]
    return canvas

def extract(row,model,pose_sha):
    if sha(row['path'])!=row['sha256']:raise ValueError('Source bytes changed')
    cap=cv2.VideoCapture(row['path']);fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps<=0 or n<=0:cap.release();raise ValueError('Unreadable source')
    spans=windows(row); requests=[frame_indices(a,b,fps,n) for a,b in spans]
    needed=sorted({int(i) for ids in requests for i in ids}); needed_set=set(needed)
    predictions={};rgbs={}; batch=[];batch_ids=[];shape=None
    def infer_batch():
        results=model.predict(batch,imgsz=640,conf=.25,iou=.45,device=0,quantize=16,verbose=False,max_det=100)
        for idx,im,result in zip(batch_ids,batch,results):
            xy=result.keypoints.data.cpu().numpy().astype(np.float32) if result.keypoints is not None else np.empty((0,17,3),np.float32)
            boxes=result.boxes.xyxy.cpu().numpy().astype(np.float32)
            scores=result.boxes.conf.cpu().numpy().astype(np.float32)
            predictions[idx]=(xy,boxes,scores);rgbs[idx]=resize_rgb(im)
        batch.clear();batch_ids.clear()
    sparse_seek=fps>120 and n>5000
    try:
        for idx in (needed if sparse_seek else range(needed[-1]+1)):
            if sparse_seek and not cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx)):
                raise ValueError(f'Could not seek to requested frame {idx}')
            ok,im=cap.read()
            if not ok:raise ValueError(f'Decode failed at frame {idx}; expected request through {needed[-1]}')
            if sparse_seek and abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(idx+1))>.1:
                raise ValueError(f'Seek position mismatch at {idx}')
            if idx not in needed_set:continue
            if shape is None:shape=list(im.shape[:2])
            batch.append(im);batch_ids.append(idx)
            if len(batch)==16:infer_batch()
        if batch:infer_batch()
    finally:cap.release()
    tracker=ByteTracker(track_thresh=.25,low_thresh=.1,track_buffer=4,min_hits=1,camera_id=row['sample_id'])
    observed={}; prior=None; gaps=[]
    for idx in needed:
        ts=idx/fps;xy,boxes,scores=predictions[idx]
        # Expire by source time, not wall time or the number of requested frames.
        tracker._tracks=[t for t in tracker.all_tracks if ts-t.last_seen<=PROTOCOL['max_gap_seconds']]
        if prior is not None and ts-prior>PROTOCOL['max_gap_seconds']:gaps.append([prior,ts])
        prior=ts
        dets=[Detection(box.tolist(),float(score),timestamp=ts+1e-9) for box,score in zip(boxes,scores)]
        tracks=tracker.update(dets);used=set(); frame=[]
        for tr in tracks:
            if tr.time_since_update:continue
            # A keypoint set is attached only to the actual matched observation.
            for j,box in enumerate(boxes):
                if j not in used and np.allclose(box,tr.bbox,atol=1e-4):
                    frame.append((tr.track_id,xy[j],box));used.add(j);break
        observed[idx]=frame
    clips=[]
    for span,ids in zip(spans,requests):
        tids=sorted({t for idx in ids for t,_,_ in observed[int(idx)]}); lookup={t:i for i,t in enumerate(tids)}
        joints=np.zeros((len(tids),32,17,3),np.float32);boxes=np.zeros((len(tids),32,4),np.float32)
        for ti,idx in enumerate(ids):
            for tid,kpt,box in observed[int(idx)]:joints[lookup[tid],ti]=kpt;boxes[lookup[tid],ti]=box
        valid=joints[...,2]>=PROTOCOL['joint_conf']
        frame_valid=(valid[:,:,5:].sum(-1)>=PROTOCOL['min_body_joints'])
        usable=frame_valid.sum(-1)>=PROTOCOL['min_valid_frames']
        clips.append(dict(start=span[0],end=span[1],frame_indices=ids.tolist(),timestamps=(ids/fps).tolist(),
                          rgb=torch.from_numpy(np.stack([rgbs[int(i)] for i in ids])),
                          keypoints=torch.from_numpy(joints),boxes=torch.from_numpy(boxes),track_ids=tids,
                          frame_valid=torch.from_numpy(frame_valid),usable=torch.from_numpy(usable),
                          quality=dict(usable_tracks=int(usable.sum()),tracks=len(tids),
                                       frame_coverage=float(frame_valid.any(0).mean()) if len(tids) else 0.,
                                       median_person_height=float(np.median((boxes[...,3]-boxes[...,1])[frame_valid])) if frame_valid.any() else 0.)))
    return dict(signature=cache_signature(row,pose_sha),sample_id=row['sample_id'],source_sha256=row['sha256'],
                pose_sha256=pose_sha,source_shape=shape,fps=fps,decoded_frames_requested=len(needed),
                sampling_gaps=gaps,clips=clips,protocol=PROTOCOL,status='complete',
                decoder='verified_frame_seek_high_rate' if sparse_seek else 'sequential')

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['gmd','tnue','fallvision','vfd']);p.add_argument('--limit',type=int);a=p.parse_args()
    offline();torch.set_num_threads(4);cv2.setNumThreads(2)
    if not POSE.is_file() or POSE.stat().st_size!=POSE_SIZE:raise ValueError('Authorized local pose weight missing')
    pose_sha=sha(POSE)
    if pose_sha!=read(OUT/'pose_provenance.json')['sha256']:raise ValueError('Pose weight changed')
    from ultralytics import YOLO
    model=YOLO(str(POSE),task='pose');manifest=read(CACHE/'manifests'/f'{a.dataset}.json')
    rows=manifest['rows'][:a.limit];dest=CACHE/a.dataset;dest.mkdir(parents=True,exist_ok=True)
    state=dict(dataset=a.dataset,status='running',expected=len(manifest['rows']),requested=len(rows),completed=0,failures=[],pose_sha256=pose_sha,manifest_sha256=sha(CACHE/'manifests'/f'{a.dataset}.json'))
    start=time.monotonic()
    for i,row in enumerate(rows):
        path=dest/(row['sample_id']+'.pt')
        try:
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=True)
                if saved['signature']!=cache_signature(row,pose_sha):raise ValueError('Incompatible cache')
            else:
                saved=extract(row,model,pose_sha);temp=path.with_suffix('.pt.tmp');torch.save(saved,temp);temp.replace(path)
            state['completed']+=1
        except Exception as exc:
            state['failures'].append(dict(sample_id=row['sample_id'],path=row['path'],error=str(exc)))
            print('extraction failure',row['sample_id'],repr(exc),flush=True)
        state['elapsed_seconds']=time.monotonic()-start
        write(OUT/f'extraction_{a.dataset}.json',state)
        if i%10==0 or i==len(rows)-1:print(a.dataset,i+1,'/',len(rows),'completed',state['completed'],'failures',len(state['failures']),flush=True)
    state['status']='complete' if state['completed']==state['expected'] else 'partial_or_failures'
    write(OUT/f'extraction_{a.dataset}.json',state)

if __name__=='__main__':main()
