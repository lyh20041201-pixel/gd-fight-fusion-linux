"""Contact sheets only; this script never infers action labels."""
from pathlib import Path
import json
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/data_review/tnue_review'

def main():
    OUT.mkdir(exist_ok=True)
    rows=json.loads((ROOT/'datasets/video_events/rebuild/tnue_audit.json').read_text())['videos']
    panels=[];index=[]
    for i,row in enumerate(rows):
        if not row['readable']:continue
        cap=cv2.VideoCapture(row['path']);frames=[]
        times=np.linspace(0,max(0,row['duration']-.1),6)
        for ts in times:
            cap.set(cv2.CAP_PROP_POS_MSEC,float(ts)*1000);ok,f=cap.read()
            if not ok:f=np.zeros((144,256,3),np.uint8)
            f=cv2.resize(f,(256,144));cv2.putText(f,f'{ts:.1f}s',(4,19),0,.55,(0,255,255),1);frames.append(f)
        cap.release()
        title=np.zeros((30,1536,3),np.uint8)
        cv2.putText(title,f'{i}: {Path(row["path"]).parent.name}/{Path(row["path"]).name}  duration {row["duration"]:.1f}s',(5,21),0,.6,(255,255,255),1)
        panels.append(np.vstack([title,np.hstack(frames)]));index.append(dict(index=i,**row))
        if len(panels)==6:
            cv2.imwrite(str(OUT/f'batch_{len(index)//6:02d}.jpg'),np.vstack(panels));panels=[]
    if panels:cv2.imwrite(str(OUT/f'batch_{len(index)//6+1:02d}.jpg'),np.vstack(panels))
    (OUT/'index.json').write_text(json.dumps(index,indent=2))
    print('reviewed metadata',len(index),flush=True)

if __name__=='__main__':main()
