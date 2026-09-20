"""Create audited temporal training manifests, keeping external tests sealed."""
from pathlib import Path
import csv, hashlib, json, re
import cv2

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'datasets/video_events/rebuild'

def video_info(p):
    c=cv2.VideoCapture(str(p));fps=c.get(cv2.CAP_PROP_FPS);n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));ok,_=c.read();c.release()
    if not ok or fps<=0 or n<=0:raise ValueError(f'Unreadable video {p}')
    return dict(fps=fps,frames=n,duration=n/fps,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def gmd():
    download=json.loads((OUT/'gmd_download.json').read_text())
    if download['errors']:raise ValueError('Incomplete GMDCSA download')
    rows=[];audit=[];errors=[]
    for subject in range(1,5):
        group=f'subject-{subject}';split={1:'train',2:'train',3:'validation',4:'test'}[subject]
        for category in ['ADL','Fall']:
            folder=ROOT/'datasets/gmdcsa24'/f'Subject {subject}'
            lines=(folder/f'{category}.csv').read_text(encoding='utf-8-sig').splitlines()[1:]
            annotations={line.split(',')[0].strip():line for line in lines if line.strip()}
            for p in sorted((folder/category).glob('*.mp4')):
                info=video_info(p);line=annotations.get(p.name)
                if line is None:errors.append(str(p));continue
                intervals=re.findall(r'Fall(?:ing)?\s*(?:\([^)]*\))?\s*\[\s*([\d.]+)\s*to\s*([\d.]+)\s*\]',line,re.I) if category=='Fall' else [('0',str(info['duration']))]
                corrections={(2,'11.mp4'):(5.9,10,'missing closing parenthesis; original timestamps retained'),
                    (4,'09.mp4'):(2.3,6,'missing to between original endpoints'),
                    (4,'15.mp4'):(1.7,info['duration'],'fall range absent; sitting annotation ends at 1.7; visual timeline confirms subsequent fall')}
                correction=None
                if category=='Fall' and (subject,p.name) in corrections:
                    a,b,correction=corrections[subject,p.name];intervals=[(a,b)]
                if not intervals:errors.append(dict(path=str(p),annotation=line));continue
                audit.append(dict(path=str(p),group=group,split=split,category=category,annotation=line,correction=correction,**info))
                for a,b in intervals:
                    start=float(a);end=min(float(b),info['duration'])
                    if end<=start:raise ValueError(f'Invalid interval {p}')
                    rows.append(dict(path=str(p),label=int(category=='Fall'),group=group,split=split,start=start,end=end,sha256=info['sha256'],annotation=line))
    (OUT/'gmd_audit.json').write_text(json.dumps(dict(videos=audit,errors=errors),indent=2))
    if errors:raise ValueError(errors)
    if len(audit)!=160:raise ValueError(f'Expected 160 videos, got {len(audit)}')
    manifest=dict(dataset='GMDCSA-24',labels=['normal','fall'],rows=rows,source_revision=download['revision'],
        protocol='subject 1+2 train; subject 3 validation; subject 4 test; FallVision external test only',
        label_semantics='Fall includes the source annotated fall/fallen interval, including bed falls; not an onset-only label')
    (OUT/'gmd_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('gmd',len(audit),'videos',len(rows),'segments',flush=True)

if __name__=='__main__':gmd()
