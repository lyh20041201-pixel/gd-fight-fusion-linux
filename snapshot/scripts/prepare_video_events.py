"""Reproducible source audit and video manifests. Never silently omit missing videos."""
from pathlib import Path
import hashlib, json, random, subprocess
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'datasets/video_events'

def write(name,data):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def airtlab():
    source=ROOT/'datasets/airtlab_source'
    files=sorted(source.glob('violence-detection-dataset/*/cam*/*.mp4'))
    if len(files)!=350:raise RuntimeError(f'AIRTLab expected 350 videos, found {len(files)}')
    groups={}
    for path in files:
        label=path.parent.parent.name
        groups.setdefault((label,path.stem),[]).append(path)
    if any(len(v)!=2 for v in groups.values()):raise ValueError('camera pair incomplete')
    split_groups={};rng=random.Random(42)
    for label in ('non-violent','violent'):
        keys=sorted(k for k in groups if k[0]==label);rng.shuffle(keys)
        a=round(len(keys)*.7);b=a+round(len(keys)*.15)
        for i,key in enumerate(keys):split_groups[key]='train' if i<a else 'validation' if i<b else 'test'
    rows=[]
    for key,paths in sorted(groups.items()):
        for path in paths:
            rows.append(dict(path=str(path),label=0 if key[0]=='non-violent' else 1,
                split=split_groups[key],group='/'.join(key),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write('airtlab.json',dict(labels=['normal','fight'],rows=rows,seed=42,
        source='https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos',
        revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()))

def omnifall():
    import pandas as pd
    api='https://huggingface.co/api/datasets/simplexsigil2/omnifall'
    r=requests.get(api,timeout=40);r.raise_for_status();rev=r.json()['sha']
    base=f'https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/{rev}/'
    folder=OUT/'omnifall_metadata';folder.mkdir(parents=True,exist_ok=True)
    for name in ('README.md','LABELS.md','STRUCTURE.md'):
        r=requests.get(base+name,timeout=40);r.raise_for_status();(folder/name).write_bytes(r.content)
    rows=[]
    for split in ('train','validation','test'):
        listing=requests.get(api+'/tree/'+rev+'/parquet/of-sta-cs',timeout=40)
        listing.raise_for_status()
        for f in listing.json():
            if not f['path'].split('/')[-1].startswith(split+'-'):continue
            r=requests.get(base+f['path'],timeout=60);r.raise_for_status()
            path=folder/Path(f['path']).name;path.write_bytes(r.content)
            for row in pd.read_parquet(path).to_dict('records'):
                video=ROOT/'datasets/omnifall'/str(row['dataset'])/'video'/(str(row['path'])+'.mp4')
                rows.append({**row,'split':split,'video':str(video),'available':video.is_file()})
    write('omnifall_audit.json',dict(source=api,revision=rev,config='of-sta-cs',
        segments=len(rows),available=sum(r['available'] for r in rows),
        missing_videos=sorted({r['video'] for r in rows if not r['available']}),rows=rows,
        status='ready' if rows and all(r['available'] for r in rows) else 'missing_source_videos'))
    if not rows:raise ValueError('No official split rows downloaded')
    if all(r['available'] for r in rows):
        manifest=[];hashes={}
        for r in rows:
            hashes.setdefault(r['video'],hashlib.sha256(Path(r['video']).read_bytes()).hexdigest())
            manifest.append(dict(path=r['video'],label=int(r['label']) if int(r['label']) in (1,2) else 0,
                start=float(r['start']),end=float(r['end']),split=r['split'],
                group=f"{r['dataset']}/subject-{r['subject']}",sha256=hashes[r['video']]))
        write('omnifall.json',dict(labels=['other','fall','fallen'],rows=manifest,revision=rev))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['airtlab','omnifall']);a=p.parse_args()
    globals()[a.dataset]()
