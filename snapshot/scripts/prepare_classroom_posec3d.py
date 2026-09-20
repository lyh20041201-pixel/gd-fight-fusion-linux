"""Acquire only the 27 Le2i lecture-room videos; keep compact H.264 copies.

This is an acted, single-room pilot, NOT a dataset of classes in progress.
OmniFall temporal labels and subject IDs define the split before training.
"""
from pathlib import Path
import sys
import concurrent.futures
import hashlib
import json
import re
import shutil
import subprocess
import time
import urllib.parse
import zipfile

import cv2
import imageio_ffmpeg
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/le2i_lecture'
OUT = ROOT / 'results/classroom_posec3d'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def metadata():
    rows = []
    for source in sorted((ROOT / 'datasets/video_events/omnifall_metadata').glob('*.parquet')):
        df = pd.read_parquet(source)
        df = df[(df.dataset == 'le2i') & df.path.str.startswith('Lecture_room/')]
        for path, group in df.groupby('path'):
            subjects = group.subject.unique()
            assert len(subjects) == 1
            subject = int(subjects[0])
            split = 'test' if source.name.startswith('test-') else ('validation' if subject == 1 else 'train')
            rows.append(dict(id=int(path.split('_')[-1]), source_path=path, subject=subject,
                split=split, source_annotation=source.name, source_annotation_sha256=sha(source),
                spans=group[['start', 'end', 'label']].sort_values('start').to_dict('records')))
    assert len(rows) == 27
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not ({r['subject'] for r in rows if r['split'] == a} & {r['subject'] for r in rows if r['split'] == b})
    return sorted(rows, key=lambda r: r['id'])


def acquire(row):
    index = row['id']
    output = DATA / 'videos' / f'video_{index:02}.mp4'
    audit = DATA / 'provenance' / f'video_{index:02}.json'
    if output.exists() and audit.exists():
        info = json.loads(audit.read_text(encoding='utf-8'))
        assert sha(output) == info['video_sha256']
        return dict(row, **info)
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = DATA / 'download_tmp' / f'video_{index:02}'
    scratch.mkdir(parents=True, exist_ok=True)
    archive = scratch / 'source.zip'
    original = scratch / 'source.avi'
    remote_name = f'Lecture_room/Lecture room/video ({index}).avi'
    url = 'https://www.kaggle.com/api/v1/datasets/download/tuyenldvn/falldataset-imvia/' + urllib.parse.quote(remote_name, safe='')
    if index == 1 and (DATA / 'raw/video_01.avi').exists():
        original = DATA / 'raw/video_01.avi'
        received = (DATA / 'raw/video_1.download').stat().st_size
    else:
        for attempt in range(3):
            try:
                with requests.get(url, stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    with archive.open('wb') as f:
                        for block in response.iter_content(1024 * 1024):
                            f.write(block)
                with zipfile.ZipFile(archive) as z:
                    entries = [n for n in z.namelist() if n.lower().endswith('.avi')]
                    assert len(entries) == 1, entries
                    with z.open(entries[0]) as src, original.open('wb') as dest:
                        shutil.copyfileobj(src, dest)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        received = archive.stat().st_size
    original_hash = sha(original)
    cap = cv2.VideoCapture(str(original))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    shape = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))]
    cap.release()
    assert fps > 0 and frames > 0 and min(shape) > 0
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-i', str(original),
        '-map', '0:v:0', '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)], check=True)
    cap = cv2.VideoCapture(str(output))
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    output_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert count == frames and abs(output_fps - fps) < .001, (index, count, frames, fps, output_fps)
    assert max(s['end'] for s in row['spans']) <= frames / fps + .2
    info = dict(video=str(output.relative_to(ROOT)), video_sha256=sha(output), source_url=url,
        source_avi_sha256=original_hash, downloaded_bytes=received, stored_bytes=output.stat().st_size,
        fps=fps, frames=frames, shape=shape, duration=frames/fps,
        transcode='H.264 CRF 18, original frame rate/count/size retained; audio discarded')
    write(audit, info)
    # Delete only files this invocation created inside this video's scratch folder.
    for temporary in (archive, original):
        if temporary.exists() and temporary.resolve().is_relative_to(scratch.resolve()):
            temporary.unlink()
    print(f"ready video_{index:02} {row['split']} {frames/fps:.1f}s {info['stored_bytes']/1e6:.2f}MB", flush=True)
    return dict(row, **info)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    rows = metadata()
    write(OUT / 'split_plan.json', dict(description='Acted Le2i lecture room; no dense class in progress',
        test='Original OmniFall held-out subject 7; only two fall events',
        validation='Subject 1 reserved before model fitting', videos=rows))
    acquired = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for future in concurrent.futures.as_completed([pool.submit(acquire, row) for row in rows]):
            acquired.append(future.result())
            write(OUT / 'manifest.partial.json', dict(videos=sorted(acquired, key=lambda r:r['id'])))
    write(OUT / 'manifest.json', dict(videos=sorted(acquired, key=lambda r:r['id']),
        scene='Le2i lecture room, single actor, simulated falls; NOT students in class',
        annotations='OmniFall README: CC-BY-NC-SA-4.0 (frontmatter says CC-BY-NC-4.0); original video rights retained by source',
        stored_bytes=sum(r['stored_bytes'] for r in acquired),
        downloaded_bytes=sum(r['downloaded_bytes'] for r in acquired)))


if __name__ == '__main__':
    main()
