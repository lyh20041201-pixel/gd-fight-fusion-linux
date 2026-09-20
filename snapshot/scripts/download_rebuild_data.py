"""Download selected original datasets; keep source inventory and explicit failures."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, hashlib, json, time
from urllib.parse import quote
import requests

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'datasets/video_events/rebuild'
AUDIT.mkdir(parents=True, exist_ok=True)

def gmd():
    tree = json.loads((AUDIT/'gmd_source.json').read_text())
    revision = tree['sha']
    rows = [r for r in tree['tree'] if r['type']=='blob']
    def fetch(row):
        dest = ROOT/'datasets/gmdcsa24'/row['path']
        dest.parent.mkdir(parents=True,exist_ok=True)
        for attempt in range(4):
            try:
                if not dest.exists() or dest.stat().st_size != row['size']:
                    url = f'https://raw.githubusercontent.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/{revision}/'+quote(row['path'])
                    r=requests.get(url,timeout=90);r.raise_for_status()
                    if len(r.content)!=row['size']:raise ValueError('Size mismatch')
                    dest.write_bytes(r.content)
                blob=dest.read_bytes()
                git_sha=hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
                if git_sha!=row['sha']:raise ValueError('Git blob hash mismatch')
                return dict(path=str(dest),sha256=hashlib.sha256(blob).hexdigest(),size=len(blob))
            except Exception:
                if attempt==3:raise
                time.sleep(2)
    result=[];errors=[]
    with ThreadPoolExecutor(6) as pool:
        jobs={pool.submit(fetch,row):row['path'] for row in rows}
        for future in as_completed(jobs):
            try:result.append(future.result())
            except Exception as e:errors.append(dict(path=jobs[future],error=str(e)))
            print('gmd',len(result),'/',len(rows),'errors',len(errors),flush=True)
    (AUDIT/'gmd_download.json').write_text(json.dumps(dict(revision=revision,files=result,errors=errors),indent=2))
    if errors:raise RuntimeError(errors)

def tnue():
    import gdown
    folders={'collected':'1NfBqK1FtkFWS5uFqplSgdEp-a89VVTud','recorded':'17o4gx61iFswj6k7pJ0Wxw34uizw6aHnb','annotations':'1KZYjg2zAkpgUZLnKFpWCPNNrqpAXsqNM'}
    rows=[]
    for name,fid in folders.items():
        files=gdown.download_folder(id=fid,output=str(ROOT/'datasets/tnue'/name),skip_download=True,quiet=True,use_cookies=False)
        rows += [dict(id=f.id,path=f.path,local_path=f.local_path,source_folder=fid) for f in files]
        print('inventory',name,len(files),flush=True)
    (AUDIT/'tnue_inventory.json').write_text(json.dumps(rows,indent=2))
    errors=[];success=[]
    def fetch(row):
        dest=Path(row['local_path']);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            for attempt in range(4):
                try:
                    url='https://drive.usercontent.google.com/download?id='+row['id']+'&export=download&confirm=t'
                    with requests.get(url,stream=True,timeout=(20,90)) as r:
                        r.raise_for_status()
                        if 'text/html' in r.headers.get('Content-Type',''):raise ValueError('Drive returned HTML instead of file')
                        partial=dest.with_suffix(dest.suffix+'.partial')
                        with partial.open('wb') as f:
                            for b in r.iter_content(1024*1024):f.write(b)
                        partial.replace(dest)
                    break
                except Exception:
                    if attempt==3:raise
                    time.sleep(2)
        return dict(**row,size=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    with ThreadPoolExecutor(3) as pool:
        jobs={pool.submit(fetch,row):row for row in rows}
        for f in as_completed(jobs):
            try:success.append(f.result())
            except Exception as e:errors.append(dict(file=jobs[f],error=str(e)))
            print('tnue',len(success),'/',len(rows),'errors',len(errors),flush=True)
    (AUDIT/'tnue_download.json').write_text(json.dumps(dict(files=success,errors=errors),indent=2))
    if errors:raise RuntimeError(f'{len(errors)} failed; see download audit')

def vfd():
    import gdown
    dest=ROOT/'datasets/vfd2000/new_youtube.zip';dest.parent.mkdir(parents=True,exist_ok=True)
    gdown.download(id='1bpP9_4pUf7ffriQIRYJhK9L-Z3lUlBxb',output=str(dest),quiet=False,use_cookies=False,resume=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['gmd','tnue','vfd']);args=p.parse_args()
    globals()[args.dataset]()
