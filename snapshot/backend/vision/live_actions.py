"""Local live inference using sealed checkpoints and training-compatible transforms.

This module has no web dependencies and runs in the original training environment.
"""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import math
import time
import cv2
import numpy as np
import torch
from .detector import Detection
from .tracker import ByteTracker
from .stgcnpp_actions import STGCNPPActionModel, prepare_stgcnpp
from .action_guards import GUARD_VERSION, assess_view, fight_people_evidence, rising_only_tracks, exclude_tracks


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def sample_live_frames(frames):
    frames = sorted({float(f[0]): f for f in frames}.values(), key=lambda f:f[0])
    if not frames:
        return [], '等待摄像头画面'
    end = frames[-1][0]
    observed_span = end-frames[0][0]
    frames = [f for f in frames if f[0] >= end-4.0]
    times = np.asarray([f[0] for f in frames])
    if observed_span < 3.5:
        return [], '正在积累4秒动作窗口'
    if len(times) < 8:
        return [], '画面帧率过低，无法判断动作；请重新接入摄像头'
    if times[-1]-times[0] < 3.5 or np.diff(times).max() > .5:
        return [], '画面中断超过0.5秒，等待连续画面'
    indices = [int(np.argmin(abs(times-t))) for t in np.linspace(times[0], times[-1],32)]
    return [frames[i] for i in indices], None


def rgb_array(images):
    out=[]
    for image in images:
        h,w=image.shape[:2]; scale=112/max(h,w)
        small=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))))
        canvas=np.zeros((112,112,3),np.uint8)
        y=(112-small.shape[0])//2; x=(112-small.shape[1])//2
        canvas[y:y+small.shape[0],x:x+small.shape[1]]=small[...,::-1]
        out.append(canvas)
    return np.stack(out)


def rgb_tensor(rgb):
    value=torch.as_tensor(rgb).float().permute(3,0,1,2)/255
    mean=torch.tensor([.43216,.394666,.37645])[:,None,None,None]
    std=torch.tensor([.22803,.22145,.216989])[:,None,None,None]
    return (value-mean)/std


class PoseWindows:
    def __init__(self, model):
        self.model=model
        self.cameras=OrderedDict()

    def detect(self, images):
        return self.model.predict(images,imgsz=640,conf=.25,iou=.45,
            device=0,half=False,verbose=False,max_det=100,save=False)

    def clip(self, camera, frames, selected):
        state=self.cameras.get(camera)
        if state is None or frames[-1][0]-state['last'] > 5 or frames[-1][0] <= state['last']-4:
            state=dict(tracker=ByteTracker(track_thresh=.25,low_thresh=.1,track_buffer=4,
                       min_hits=1,camera_id=camera),observed={},last=-float('inf'))
            self.cameras[camera]=state
        self.cameras.move_to_end(camera)
        while len(self.cameras)>8: self.cameras.popitem(last=False)
        pending=[f for f in frames if f[0]>state['last']]
        for begin in range(0,len(pending),8):
            batch=pending[begin:begin+8]
            results=self.detect([f[1] for f in batch])
            for frame,result in zip(batch,results):
                ts=frame[0]
                joints=result.keypoints.data.cpu().numpy().astype(np.float32)
                boxes=result.boxes.xyxy.cpu().numpy().astype(np.float32)
                scores=result.boxes.conf.cpu().numpy().astype(np.float32)
                tracker=state['tracker']
                tracker._tracks=[t for t in tracker.all_tracks if ts-t.last_seen<=.5]
                dets=[Detection(box.tolist(),float(score),timestamp=ts) for box,score in zip(boxes,scores)]
                tracks=tracker.update(dets); used=set(); observations=[]
                for track in tracks:
                    if track.time_since_update: continue
                    for j,box in enumerate(boxes):
                        if j not in used and np.allclose(box,track.bbox,atol=1e-4):
                            observations.append((track.track_id,joints[j],box));used.add(j);break
                state['observed'][ts]=observations;state['last']=ts
        state['observed']={ts:v for ts,v in state['observed'].items() if ts>=frames[-1][0]-6}
        ids=sorted({tid for ts, *_ in selected for tid,_,_ in state['observed'].get(ts,[])})
        if len(ids)>32: raise ValueError('同窗人体轨迹过多，暂停骨架判断')
        lookup={tid:i for i,tid in enumerate(ids)}
        joints=np.zeros((len(ids),len(selected),17,3),np.float32)
        boxes=np.zeros((len(ids),len(selected),4),np.float32)
        for i,(ts,*_) in enumerate(selected):
            for tid,points,box in state['observed'].get(ts,[]):
                joints[lookup[tid],i]=points;boxes[lookup[tid],i]=box
        # Relative seconds avoid loss of sub-second resolution at Unix timestamps.
        first=selected[0][0]
        unique={ts:i for i,ts in enumerate(sorted({f[0] for f in selected}))}
        return dict(keypoints=torch.from_numpy(joints),boxes=torch.from_numpy(boxes),
            frame_indices=[unique[f[0]] for f in selected],
            timestamps=[f[0]-first for f in selected],track_ids=ids)


class LiveActionModels:
    fall_kind='stgcnpp'

    def __init__(self, manifest_path):
        self.manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        if not torch.cuda.is_available(): raise RuntimeError('动作识别需要已验证的CUDA环境')
        torch.set_num_threads(4);cv2.setNumThreads(2)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        for spec in [self.manifest['fall'],*self.manifest['fight']['models'],self.manifest['pose']]:
            if sha256(spec['path'])!=spec['sha256']: raise ValueError('动作模型或姿态权重校验失败')
        spec=self.manifest['fall']
        self.fall_kind=spec.get('architecture','stgcnpp')
        self.fall_details={}
        if self.fall_kind=='safer_posec3d':
            from .posec3d_actions import load_posec3d
            self.fall=load_posec3d(spec['path'])
            if not 0 < spec['threshold'] <= 1: raise ValueError('跌倒阈值无效')
        elif self.fall_kind=='stgcnpp':
            saved=torch.load(spec['path'],map_location='cpu',weights_only=True)
            self.fall=STGCNPPActionModel('fall'); self.fall.load_state_dict(saved['state_dict'])
            self.fall=self.fall.cuda().eval()
            if saved['threshold']!=spec['threshold']: raise ValueError('跌倒阈值与封存模型不一致')
        else:
            raise ValueError('未知跌倒模型架构')
        from torchvision.models.video import r3d_18
        self.fight=[]
        for spec in self.manifest['fight']['models']:
            saved=torch.load(spec['path'],map_location='cpu',weights_only=True)
            model=r3d_18(weights=None);model.fc=torch.nn.Linear(512,2)
            model.load_state_dict(saved['state_dict']);self.fight.append(model.cuda().eval())
        from ultralytics import YOLO
        self.pose=PoseWindows(YOLO(self.manifest['pose']['path'],task='pose'))
        self.version=sha256(manifest_path)

    def pose_preview(self, frame):
        # Same pose weights/options, without advancing the action-window tracker.
        result=self.pose.detect([frame[1]])[0]
        return dict(timestamp=frame[0],keypoints=result.keypoints.data.cpu().numpy().tolist())

    def score_fall_clip(self, clip, shape):
        self.fall_details={}
        if self.fall_kind=='safer_posec3d':
            from .posec3d_actions import score_posec3d
            self.fall_details=score_posec3d(self.fall,clip,shape,self.manifest['fall']['threshold'])
            return self.fall_details['score']
        item=prepare_stgcnpp(clip,shape)
        with torch.inference_mode(),torch.autocast('cuda'):
            value=self.fall({k:v.cuda() for k,v in item.items()})
            return float(value.float().sigmoid()) if value is not None else None

    def score_rgb_clip(self, rgb):
        scores=[]
        with torch.inference_mode(),torch.autocast('cuda'):
            value=rgb_tensor(rgb).unsqueeze(0).cuda()
            for model in self.fight:
                # Match the cached-layer3 training boundary and fp16 score path.
                z=model.layer3(model.layer2(model.layer1(model.stem(value))))
                z=model.avgpool(model.layer4(z.half().float())).flatten(1)
                logits=model.fc(z)
                scores.append(float((logits[0,1]-logits[0,0]).sigmoid()))
        method=self.manifest['fight']['method']
        score=float(np.median(scores)) if method=='median' else float(np.mean(scores))
        return score

    def predict(self,camera,frames):
        started=time.monotonic()
        selected,reason=sample_live_frames(frames)
        if not selected:
            return dict(state='warming',reason=reason,actions=[],inference_ms=0)
        shape=selected[0][1].shape[:2]
        if any(f[1].shape[:2]!=shape for f in frames):
            return dict(state='warming',reason='画面尺寸变化，等待新窗口',actions=[],inference_ms=0)
        quality=assess_view(selected)
        if not quality['ok']:
            self.pose.cameras.pop(camera,None)
            return dict(state='blocked',reason=quality['reason'],actions=[],quality=quality,
                inference_ms=round((time.monotonic()-started)*1000,1))
        clip=self.pose.clip(camera,frames,selected)
        # PoseC3D receives all observed poses, then samples its own short windows.
        # Fight retains its original 4-second / 32-frame input.
        fall_source=clip
        if self.fall_kind=='safer_posec3d':
            dense=sorted({float(f[0]):f for f in frames if f[0]>=selected[-1][0]-4}.values(),key=lambda f:f[0])
            fall_source=self.pose.clip(camera,frames,dense)
        rising=rising_only_tracks(fall_source)
        fall_clip=exclude_tracks(fall_source,rising)
        fight_people=fight_people_evidence(clip)
        actions=[]
        for name,event_type in [('fall','person_fall'),('fight','suspected_fight')]:
            spec=self.manifest[name]
            action=dict(kind=name,label='跌倒' if name=='fall' else '打架',event_type=event_type,
                threshold=spec['threshold'],model=spec['label'],score=None,state='unknown',
                reason=None,start=selected[0][0],end=selected[-1][0],model_version=self.version,
                guard_version=GUARD_VERSION)
            try:
                if name=='fall':
                    score=self.score_fall_clip(fall_clip,shape)
                    if self.fall_kind=='safer_posec3d':
                        action.update({k:v for k,v in self.fall_details.items() if k!='probabilities'})
                    else:
                        action['usable_tracks']=int(prepare_stgcnpp(fall_clip,shape)['features'].shape[0])
                    action['rising_tracks_excluded']=sum(rising)
                    if score is None and any(rising):
                        action['reason']='观测到起身，未发现此前下落过程；跌倒证据不足'
                else:
                    action['people_evidence']=fight_people
                    if not fight_people['eligible']:
                        score=None
                        action['reason']=fight_people['reason']
                    else:
                        score=self.score_rgb_clip(rgb_array([f[1] for f in selected]))
                if score is not None and not math.isfinite(score): raise ValueError('模型分数无效')
                action.update(score=score,state='unknown' if score is None else 'candidate' if score>=spec['threshold'] else 'normal')
                if name=='fall' and self.fall_kind=='safer_posec3d' and score is not None and not self.fall_details['is_fall']:
                    action['state']='normal'
                if score is None and action['reason'] is None: action['reason']='未获得至少8个有效时刻的人体骨架'
            except Exception as exc:
                import logging
                logging.getLogger(__name__).exception('Live action inference failed: %s',name)
                action.update(state='error',reason='模型推理失败，请查看本地服务日志')
            actions.append(action)
        return dict(state='running',actions=actions,inference_ms=round((time.monotonic()-started)*1000,1),
            unique_frames=len({f[0] for f in selected}),window_start=selected[0][0],window_end=selected[-1][0],
            quality=quality,guard_version=GUARD_VERSION)
