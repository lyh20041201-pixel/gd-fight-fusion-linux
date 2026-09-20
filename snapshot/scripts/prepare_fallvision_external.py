"""Extract only original RGB archives and seal the FallVision external set."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import argparse,hashlib,json,subprocess
from prepare_rebuild_manifests import video_info

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'datasets/fallvision'
AUDIT=ROOT/'datasets/video_events/rebuild'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extract-only',action='store_true');a=parser.parse_args()
    archives=sorted((BASE/'archives').rglob('*.rar'))
    raw=[p for p in archives if 'Raw Video' in p.parts]
    done_path=AUDIT/'fallvision_extraction.json'
    done=json.loads(done_path.read_text()) if done_path.exists() else {}
    for archive in raw:
        key=archive.relative_to(BASE/'archives').as_posix()
        if key in done:continue
        destination=BASE/'raw'/archive.relative_to(BASE/'archives').parent/archive.stem
        destination.mkdir(parents=True,exist_ok=True)
        listed=subprocess.run(['tar','-tf',str(archive)],capture_output=True,text=True,check=True)
        names=listed.stdout.splitlines()
        for name in names:
            if not (destination/name).resolve().is_relative_to(destination.resolve()):raise ValueError(name)
        extracted=subprocess.run(['tar','-xf',str(archive),'-C',str(destination)],capture_output=True,text=True)
        if extracted.returncode:raise RuntimeError(extracted.stderr)
        done[key]=dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),entry_count=len(names),destination=str(destination))
        done_path.write_text(json.dumps(done,indent=2))
        print('extracted',key,len(names),flush=True)
    if a.extract_only:return
    # Public release comprises ten fall and ten nonfall original-video archives.
    if len(raw)!=20 or len(done)!=20:raise ValueError(f'Incomplete raw archive set: {len(raw)} archives, {len(done)} extracted')
    files=sorted((BASE/'raw').rglob('*.mp4'));errors=[];by_hash={}
    def inspect(path):
        try:return path,video_info(path),None
        except Exception as exc:return path,None,str(exc)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(path,info,error) in enumerate(pool.map(inspect,files)):
            if error:errors.append(dict(path=str(path),error=error));continue
            if 'No Fall' in path.parts:label=0
            elif 'Fall' in path.parts:label=1
            else:raise ValueError('Unknown source label: '+str(path))
            row=dict(path=str(path),label=label,split='external_test',group=path.parent.name,
                     sha256=info['sha256'],duration=info['duration'],source_label='No Fall' if label==0 else 'Fall')
            by_hash.setdefault(info['sha256'],[]).append(row)
            if i%100==0:print('inspect',i+1,len(files),flush=True)
    conflicts={sha:rs for sha,rs in by_hash.items() if len({r['label'] for r in rs})>1}
    gmd=json.loads((AUDIT/'gmd_manifest.json').read_text());overlap=set(by_hash)&{r['sha256'] for r in gmd['rows']}
    if overlap:raise ValueError('Exact GMD/FallVision overlap: '+str(overlap))
    rows=[rs[0] for sha,rs in by_hash.items() if sha not in conflicts]
    manifest=dict(dataset='FallVision',source='doi:10.7910/DVN/75QPKK',labels=['normal','fall'],rows=rows,
                  raw_file_count=len(files),unique_samples=len(rows),class_counts=dict(Counter(r['label'] for r in rows)),
                  errors=errors,duplicates=[rs for rs in by_hash.values() if len(rs)>1],
                  conflicting_label_hashes_excluded=sorted(conflicts),exact_gmd_overlap=sorted(overlap),
                  protocol='All available raw RGB videos, source folder labels; no masked/keypoint copies; external evaluation only',
                  grouping_limitation='Archive grouping is provenance only; subject identifiers are not inferred from filenames')
    (AUDIT/'fallvision_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(dict(files=len(files),unique=len(rows),classes=manifest['class_counts'],errors=len(errors),conflicts=len(conflicts)),flush=True)

if __name__=='__main__':main()
