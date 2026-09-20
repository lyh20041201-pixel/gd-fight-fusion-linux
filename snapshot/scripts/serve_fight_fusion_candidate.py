"""Explicit candidate named-pipe/replay worker; never starts the production service."""
from pathlib import Path
from collections import deque
import argparse
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.update(YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false')
import cv2
import numpy as np
import torch
from backend.vision.fight_fusion_live import FightFusionLiveAdapter


def replay(adapter, video, output):
    cap=cv2.VideoCapture(str(video))
    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps<=0:
        raise ValueError('Invalid replay video')
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError('Use a new replay output, not an overwrite')
    pending=deque();index=0;last=-float('inf');timings=[];count=0
    try:
        with output.open('x',encoding='utf-8') as handle:
            while True:
                ok,image=cap.read()
                if not ok:
                    break
                timestamp=index/fps;index+=1
                pending.append((timestamp,image))
                while pending and pending[0][0]<timestamp-4.25:
                    pending.popleft()
                if timestamp-last<2:
                    continue
                last=timestamp
                result=adapter.predict(str(video),list(pending))
                handle.write(json.dumps(dict(timestamp=timestamp,**result),ensure_ascii=False,allow_nan=False)+'\n')
                handle.flush();count+=1
                if result['state']=='running':
                    timings.append(result['inference_ms'])
                print('REPLAY',round(timestamp,2),result['state'],result.get('inference_ms'),flush=True)
    finally:
        cap.release()
    summary=dict(video=str(video),predictions=count,source_seconds=index/fps,
                 inference_ms_p50=float(np.median(timings)) if timings else None,
                 inference_ms_p95=float(np.percentile(timings,95)) if timings else None,
                 peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                 limitations='Local causal file replay, not operating classroom-camera acceptance; short files may never fill the 4-second live buffer.')
    output.with_suffix('.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pipe')
    group.add_argument('--video')
    parser.add_argument('--output')
    args=parser.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    adapter=FightFusionLiveAdapter(args.manifest)
    if args.video:
        if not args.output:
            parser.error('--video requires a new --output JSONL path')
        print(json.dumps(replay(adapter,args.video,args.output),ensure_ascii=False),flush=True)
        return
    from multiprocessing.connection import Listener
    with Listener(args.pipe,family='AF_PIPE',authkey=bytes.fromhex(os.environ.pop('GD_ACTION_AUTH'))) as listener:
        with listener.accept() as connection:
            connection.send(dict(ready=True,torch=str(torch.__version__),model_version=adapter.version,
                                 fight_model=adapter.config.get('label','three-stream candidate'),candidate_only=True))
            while True:
                try:request=connection.recv()
                except EOFError:break
                if request.get('op')=='stop':break
                try:
                    result=(adapter.pose_preview(request['frames'][0]) if request.get('op')=='pose_preview'
                            else adapter.predict(request['camera'],request['frames']))
                    connection.send(result)
                except Exception as exc:
                    connection.send(dict(state='error',reason=str(exc)[:250],actions=[]))


if __name__=='__main__':main()
