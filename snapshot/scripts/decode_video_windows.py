"""Decode a video's requested frames once; preserve the original sampling math."""
import cv2,numpy as np
from backend.vision.video_events import clip_tensor

def decode_windows(row,intervals):
    cap=cv2.VideoCapture(row['path']);fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps<=0 or n<=0:cap.release();raise ValueError('Unreadable '+row['path'])
    requests=[];needed=set()
    for start,end in intervals:
        end=min(end,n/fps);start=max(start,(start+end)/2-2);end=min(end,start+4)
        ids=np.linspace(round(start*fps),max(round(start*fps),min(n-1,round(end*fps)-1)),16).astype(int)
        requests.append((ids,dict(start=start,end=end,valid_duration=end-start)));needed.update(int(i) for i in ids)
    cached={}
    try:
        for idx in range(max(needed)+1):
            ok,image=cap.read()
            if not ok:raise ValueError(f'Decode failed {row["path"]} frame {idx}')
            if idx not in needed:continue
            h,w=image.shape[:2];scale=112/max(h,w)
            resized=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))))
            canvas=np.zeros((112,112,3),np.uint8);y,x=(112-resized.shape[0])//2,(112-resized.shape[1])//2
            canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized;cached[idx]=canvas
    finally:cap.release()
    for ids,window in requests:yield clip_tensor([cached[int(i)] for i in ids]),window
