"""Perceptual-frame duplicate screening, independent of model predictions."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'datasets/video_events/rebuild'

def hashes(row):
    cap=cv2.VideoCapture(row['path']);fps=cap.get(cv2.CAP_PROP_FPS);n=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps<=0:cap.release();return []
    start=row.get('start',0);end=min(row.get('end',n/fps),n/fps-.25);result=[]
    for ts in np.linspace(start,max(start,end),5):
        cap.set(cv2.CAP_PROP_POS_MSEC,float(ts)*1000);ok,f=cap.read()
        if not ok:continue
        gray=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
        # Remove uniform white/black embedding margins, retaining full action area.
        ys=np.flatnonzero(gray.std(1)>5);xs=np.flatnonzero(gray.std(0)>5)
        if len(ys)<8 or len(xs)<8:continue
        gray=gray[ys[0]:ys[-1]+1,xs[0]:xs[-1]+1]
        small=cv2.resize(gray,(32,32)).astype(np.float32)
        dct=cv2.dct(small)[:8,:8].flatten();bits=dct>np.median(dct[1:]);bits[0]=False
        value=int.from_bytes(np.packbits(bits).tobytes(),'little')
        result.append(dict(path=row['path'],time=float(ts),hash=value))
    cap.release();return result

def main():
    train=json.loads((OUT/'tnue_manifest.json').read_text())['rows']
    external=json.loads((OUT/'vfd_manifest.json').read_text())['rows']
    a=[];b=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(hashes,train):a.extend(result)
        for i,result in enumerate(pool.map(hashes,external)):
            b.extend(result)
            if i%100==0:print('external fingerprints',i+1,len(external),flush=True)
    (OUT/'tnue_vfd_fingerprints.json').write_text(json.dumps(dict(tnue=a,vfd=b)))
    values=np.array([r['hash'] for r in b],dtype=np.uint64)
    pairs={}
    for row in a:
        xor=np.bitwise_xor(values,np.uint64(row['hash']))
        distances=np.unpackbits(xor.view(np.uint8).reshape(-1,8),axis=1).sum(1)
        for idx in np.flatnonzero(distances<=12):
            other=b[int(idx)];key=(row['path'],other['path']);distance=int(distances[idx])
            if key not in pairs or distance<pairs[key]['distance']:
                pairs[key]=dict(tnue=row,vfd=other,distance=distance)
    record=dict(status='screening_complete',method='Five frame pHashes per selected TNUE window and per VFD video; uniform margin trim; 64-bit Hamming distance <= 12',
                limitation='Candidate screen only; cannot rule out re-edited or transformed copies missed by sampled frames',
                tnue_fingerprints=len(a),vfd_fingerprints=len(b),candidates=sorted(pairs.values(),key=lambda r:r['distance']))
    (OUT/'tnue_vfd_perceptual_overlap.json').write_text(json.dumps(record,indent=2))
    print('candidate source pairs',len(pairs),flush=True)

if __name__=='__main__':main()
