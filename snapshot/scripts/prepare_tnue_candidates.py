"""Materialize AI-selected candidate windows for explicit visual review."""
from pathlib import Path
import json
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/data_review/tnue_windows'
# index: (candidate normal start, candidate fight start). NOT accepted labels yet.
CHOICES={
0:(None,4),1:(74,14),2:(0,91),3:(None,20),4:(None,18),5:(0,14),
6:(0,218),7:(None,22),8:(70,30),9:(None,24),10:(0,88),11:(0,20),
12:(None,0),13:(None,12),14:(0,94),15:(0,16),16:(0,30),17:(50,100),
18:(0,20),19:(None,20),20:(88,54),21:(None,0),22:(86,18),
24:(0,10),25:(0,17),26:(0,20),27:(70,14),28:(6,26),29:(None,5),
30:(0,5),31:(0,18),32:(None,14),33:(0,17),34:(0,10),35:(0,18),
36:(0,17),37:(None,14),38:(0,28),39:(0,23),40:(0,100),41:(None,4),
42:(58,14),43:(34,7),44:(0,36),45:(0,24),46:(0,24),
51:(0,26),52:(None,5),53:(None,2),55:(0,22),56:(0,20),58:(0,14),
59:(0,24),60:(None,0),61:(48,10),62:(None,1),63:(None,140),64:(None,0),
65:(8,2),66:(0,64),67:(None,8),68:(0,26),69:(0,14),70:(0,31),
71:(10,30),72:(0,40),75:(29,7),76:(0,16),77:(34,12),79:(0,20),
80:(None,4),81:(0,9),82:(None,6),83:(0,12),84:(0,7),85:(None,6),
86:(None,8),87:(None,10),88:(0,5),89:(None,6),90:(None,2),91:(None,3),
92:(0,4),93:(None,3),94:(0,12)}
EXCLUDED={23:'Overview insufficient to locate an unambiguous fight window',
47:'Low-detail screen recording; action ambiguous in overview',
48:'Street incident ambiguous; no clear interpersonal fighting in overview',
49:'Multi-incident compilation; source overlap cannot be reliably resolved',
50:'Heavily obstructed distant crowd; action ambiguous in overview',
54:'Very small screen-recorded action; ambiguous overview',
57:'Source file cannot decode; retry identical SHA256',
73:'Multi-incident compilation; source overlap cannot be reliably resolved',
74:'Visual duplicate of video438.mp4 (index 61); excluded even if file hash differs',
78:'Participants largely outside camera framing'}

def main():
    OUT.mkdir(exist_ok=True)
    source=json.loads((ROOT/'datasets/video_events/rebuild/tnue_audit.json').read_text())['videos']
    rows=[];panels=[]
    for idx,(normal,fight) in CHOICES.items():
        r=source[idx]
        for label,start in enumerate((normal,fight)):
            if start is None:continue
            end=min(start+4,r['duration']-.25)
            # Source header claims 176 frames; sequential decode yields only 161.
            if idx==53:end=min(end,5.2)
            if idx==62:end=4
            if idx in (84,88,92) and label==0:end=3
            cap=cv2.VideoCapture(r['path']);fps=r['fps'];n=int(r['frames'])
            ids=np.linspace(round(start*fps),max(round(start*fps),min(n-1,round(end*fps)-1)),16).astype(int)
            images=[]
            for frame in ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame));ok,f=cap.read()
                if not ok:raise ValueError((r['path'],frame))
                f=cv2.resize(f,(256,144));cv2.putText(f,f'{frame/fps:.2f}s',(3,19),0,.5,(0,255,255),1);images.append(f)
            cap.release()
            sample=f'{idx:02d}_{label}';path=OUT/f'{sample}.jpg'
            cv2.imwrite(str(path),np.vstack([np.hstack(images[j:j+4]) for j in range(0,16,4)]))
            title=np.zeros((28,1536,3),np.uint8)
            cv2.putText(title,f'{sample} {Path(r["path"]).name} CANDIDATE {"fight" if label else "normal"} {start:.1f}-{end:.1f}s',(5,20),0,.6,(255,255,255),1)
            panels.append(np.vstack([title,np.hstack([cv2.resize(images[j],(192,108)) for j in (0,2,4,6,9,11,13,15)])]))
            rows.append(dict(id=sample,source_index=idx,path=r['path'],start=start,end=end,label=label,
                             sha256=r['sha256'],evidence=str(path),status='candidate_unreviewed'))
            if len(panels)==8:
                cv2.imwrite(str(OUT/f'batch_{len(rows)//8:02d}.jpg'),np.vstack(panels));panels=[]
    if panels:cv2.imwrite(str(OUT/f'batch_{len(rows)//8+1:02d}.jpg'),np.vstack(panels))
    (OUT/'candidates.json').write_text(json.dumps(dict(rows=rows,excluded_sources=EXCLUDED),indent=2))
    print('candidate windows',len(rows),flush=True)

if __name__=='__main__':main()
