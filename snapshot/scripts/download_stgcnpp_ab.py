"""Download only the user-authorized, pinned ST-GCN++ code and NTU60 2D weights."""
from pathlib import Path
import hashlib, json, urllib.request, zipfile, time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/video_events/skeleton_stgcnpp_ab'
COMMIT = 'f2bf3a6b08e2e8dec744692d64efdb187fd6719a'
ASSETS = [
    (f'https://codeload.github.com/kennymckormick/pyskl/zip/{COMMIT}', ROOT/'third_party'/f'pyskl_{COMMIT}.zip'),
    ('https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_hrnet/j.pth', ROOT/'models/stgcnpp/ntu60_xsub_hrnet_joint.pth'),
]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records=[]
    for url,path in ASSETS:
        path.parent.mkdir(parents=True, exist_ok=True)
        reused=path.exists()
        if not reused:
            temp=path.with_suffix(path.suffix+'.part')
            count=0
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'GD-STGCNPP-AB'}),timeout=60) as response, temp.open('wb') as dst:
                while block:=response.read(1024*1024):
                    count+=len(block)
                    if count>30*1024*1024: raise ValueError('Unexpected asset size')
                    dst.write(block)
            temp.replace(path)
        content=path.read_bytes()
        record=dict(url=url,path=str(path),bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),reused=reused)
        records.append(record); print(json.dumps(record),flush=True)
    code=ASSETS[0][1]
    destination=ROOT/'third_party'/f'pyskl_{COMMIT}'
    if not destination.exists():
        destination.mkdir()
        with zipfile.ZipFile(code) as archive:
            for item in archive.infolist():
                relative=Path(*Path(item.filename).parts[1:])
                target=(destination/relative).resolve()
                if not target.is_relative_to(destination.resolve()):raise ValueError('Unsafe archive member')
                if item.is_dir():target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(archive.read(item))
    record=dict(commit=COMMIT,assets=records,source_directory=str(destination),
                downloaded_at=time.strftime('%Y-%m-%d %H:%M:%S'),datasets_downloaded=False,dependencies_installed=False)
    receipt=OUT/'download_receipt.json'
    if not receipt.exists():receipt.write_text(json.dumps(record,indent=2),encoding='utf-8')
    else:
        prior=json.loads(receipt.read_text(encoding='utf-8'))
        assert [(a['path'],a['sha256']) for a in prior['assets']]==[(a['path'],a['sha256']) for a in records]
    print('DOWNLOADS_COMPLETE',sum(r['bytes'] for r in records),flush=True)

if __name__=='__main__': main()
