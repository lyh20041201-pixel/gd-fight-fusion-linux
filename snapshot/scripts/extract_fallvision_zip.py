"""Verify the public release ZIP and extract only its twenty raw-video RARs."""
from pathlib import Path
import hashlib,json,shutil,zipfile,zlib
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'datasets/fallvision'

def main():
    archive=BASE/'dataverse_files.zip';destination=(BASE/'archives').resolve()
    records=[]
    with zipfile.ZipFile(archive) as zip:
        selected=[i for i in zip.infolist() if '/Raw Video/' in i.filename and i.filename.endswith('.rar')]
        if len(selected)!=20:raise ValueError(f'Expected twenty original-video archives, found {len(selected)}')
        for info in selected:
            path=(destination/info.filename).resolve()
            if not path.is_relative_to(destination):raise ValueError(info.filename)
            valid=False
            if path.exists() and path.stat().st_size==info.file_size:
                crc=0
                with path.open('rb') as source:
                    for block in iter(lambda:source.read(1024*1024),b''):crc=zlib.crc32(block,crc)
                valid=crc==info.CRC
            if not valid:
                path.parent.mkdir(parents=True,exist_ok=True)
                with zip.open(info) as source,path.open('wb') as output:shutil.copyfileobj(source,output,1024*1024)
            records.append(dict(name=info.filename,size=info.file_size,crc32=info.CRC))
            print('CRC verified',info.filename,flush=True)
    with archive.open('rb') as source:sha=hashlib.file_digest(source,'sha256').hexdigest()
    record=dict(source='doi:10.7910/DVN/75QPKK',downloaded='2026-09-13',zip_size=archive.stat().st_size,zip_sha256=sha,raw_archives=records)
    (ROOT/'datasets/video_events/rebuild/fallvision_download.json').write_text(json.dumps(record,indent=2))
    print('complete',archive.stat().st_size,sha,flush=True)

if __name__=='__main__':main()
