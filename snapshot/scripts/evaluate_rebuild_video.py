"""Sealed cross-dataset video evaluation; no threshold tuning on external data."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,hashlib,json,time
import torch,numpy as np,cv2
from torchvision.models.video import r3d_18
from scripts.train_video_events import decode,metrics
from scripts.decode_video_windows import decode_windows

def windows(row):
    duration=float(row['duration'])
    if duration<=4:return [(0.,duration)]
    starts=list(np.arange(0,duration-4,2.))
    starts.append(duration-4)
    return [(float(s),float(s+4)) for s in starts]

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    data=json.loads(Path(a.manifest).read_text(encoding='utf-8'));rows=data['rows']
    if any(r['split']!='external_test' for r in rows):raise ValueError('Only sealed external test manifests')
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if data['labels']!=checkpoint['labels']:raise ValueError('Label mismatch')
    model=r3d_18(weights=None);model.fc=torch.nn.Linear(model.fc.in_features,len(data['labels']));model.load_state_dict(checkpoint['state_dict']);model.cuda().eval();torch.set_num_threads(4)
    record=dict(status='running',dataset=data['dataset'],checkpoint=str(Path(a.checkpoint).resolve()),
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),threshold=checkpoint['threshold'],
        protocol='Full video: up-to-4s windows, 2s stride plus final tail window, 16 frames each; video score is max positive window score; fixed internal-validation threshold, no external tuning',
        labels=data['labels'],expected_samples=len(rows),failures=[],predictions=[])
    def save():
        (out/'evaluation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    save();t=time.perf_counter();evidence={}
    for i,row in enumerate(rows):
        try:
            scored=[]
            for clip,window in decode_windows(row,windows(row)):
                with torch.inference_mode(),torch.autocast('cuda'):
                    probs=model(clip.unsqueeze(0).float().cuda()).softmax(1)[0].cpu().tolist()
                scored.append(dict(window=window,probabilities=probs))
            peak=max(scored,key=lambda item:item['probabilities'][1])
            probs=peak['probabilities'];window=peak['window']
            prediction=int(probs[1]>=checkpoint['threshold']);truth=row['label']
            r=dict(**row,window=window,probabilities=probs,all_windows=scored,prediction=prediction,correct=prediction==truth)
            key=f'truth{truth}_pred{prediction}'
            if evidence.get(key,0)<3:
                c=cv2.VideoCapture(row['path']);images=[]
                for ts in np.linspace(window['start'],max(window['start'],window['end']-.1),8):
                    c.set(cv2.CAP_PROP_POS_MSEC,float(ts)*1000);ok,f=c.read()
                    if not ok:continue
                    f=cv2.resize(f,(320,180));cv2.putText(f,f'{ts:.2f}s',(5,20),0,.5,(0,255,255),1);images.append(f)
                c.release()
                if len(images)==8:
                    name=f'{key}_{evidence.get(key,0)}.jpg';cv2.imwrite(str(out/name),np.vstack([np.hstack(images[:4]),np.hstack(images[4:])]))
                    r['evidence']=str((out/name).resolve());evidence[key]=evidence.get(key,0)+1
            record['predictions'].append(r)
        except Exception as exc:record['failures'].append(dict(path=row['path'],error=str(exc)))
        if i%25==0:save();print('evaluate',i+1,len(rows),'failures',len(record['failures']),flush=True)
    pred=record['predictions'];record['metrics']=metrics([r['label'] for r in pred],[r['prediction'] for r in pred],2)
    record['metrics']['accuracy']=float(np.mean([r['correct'] for r in pred])) if pred else None
    record['elapsed_seconds']=time.perf_counter()-t;record['status']='complete' if not record['failures'] else 'incomplete_decode_failures';save()
    print(record['status'],record['metrics'],flush=True)

if __name__=='__main__':main()
