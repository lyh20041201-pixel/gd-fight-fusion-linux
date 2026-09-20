"""Download the two author-hosted fight datasets with bounded, resumable I/O."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
import threading
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/fight_fusion_v1'
OUT = ROOT / 'results/fight_fusion_v1/data'
SCFD_COMMIT = 'ff83b7c521e5ca4eb67a212cc42dbc54f39d6290'
SCFD_URL = f'https://codeload.github.com/seymanurakti/fight-detection-surv-dataset/zip/{SCFD_COMMIT}'
UBI_URL = 'https://socia-lab.di.ubi.pt/EventDetection/UBI_FIGHTS.zip'
UBI_BYTES = 8117112419
FLOOR_BYTES = 100 * 1024**3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    # Windows readers can briefly hold a handle without FILE_SHARE_DELETE.
    for attempt in range(30):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(0.1)


def check_disk(path: Path, pending: int = 0) -> None:
    if shutil.disk_usage(path).free - pending < FLOOR_BYTES:
        raise RuntimeError('Disk guard: operation would cross the 100 GiB free-space floor')


def check_content_range(value: str | None, start: int, end: int, total: int) -> None:
    if value != f'bytes {start}-{end}/{total}':
        raise RuntimeError(f'Unexpected Content-Range {value!r}, expected {start}-{end}/{total}')


def guarded_copy(source, target, target_directory: Path) -> None:
    while True:
        block = source.read(1024**2)
        if not block:
            return
        check_disk(target_directory, len(block))
        target.write(block)


def safe_extract(archive: Path, destination: Path) -> list[dict]:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    receipts = []
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        check_disk(root, sum(i.file_size for i in infos))
        for info in infos:
            name = PurePosixPath(info.filename.replace('\\', '/'))
            if name.is_absolute() or '..' in name.parts or any(':' in x for x in name.parts):
                raise ValueError(f'Unsafe zip member: {info.filename}')
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f'Zip symlinks are not allowed: {info.filename}')
            target = root.joinpath(*name.parts).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f'Zip member escapes destination: {info.filename}')
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(target.suffix + '.extracting')
            with z.open(info) as source, temp.open('wb') as dest:
                guarded_copy(source, dest, root)
            if temp.stat().st_size != info.file_size:
                raise ValueError(f'Extracted size mismatch: {info.filename}')
            temp.replace(target)
            receipts.append({'path': str(target), 'size': info.file_size, 'sha256': sha256(target)})
    return receipts


def download_ubi(archive: Path, progress) -> None:
    """The exception is authorized ONLY for the exact public author ZIP URL."""
    partial = archive.with_suffix('.zip.part')
    if archive.exists():
        if archive.stat().st_size != UBI_BYTES:
            raise ValueError('Existing UBI archive has unexpected length')
        return
    partial.parent.mkdir(parents=True, exist_ok=True)
    check_disk(partial.parent, UBI_BYTES - (partial.stat().st_size if partial.exists() else 0))
    session = requests.Session()
    while (partial.stat().st_size if partial.exists() else 0) < UBI_BYTES:
        start = partial.stat().st_size if partial.exists() else 0
        end = min(start + 32 * 1024**2, UBI_BYTES) - 1
        for attempt in range(5):
            try:
                # No redirect is followed with the narrowly scoped TLS exception.
                with session.get(UBI_URL, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                                 stream=True, timeout=(20, 120), verify=False, allow_redirects=False) as r:
                    if r.status_code != 206:
                        raise RuntimeError(f'UBI range returned HTTP {r.status_code}')
                    check_content_range(r.headers.get('Content-Range'), start, end, UBI_BYTES)
                    received = 0
                    with partial.open('ab') as f:
                        for chunk in r.iter_content(1024**2):
                            if not chunk:
                                continue
                            if received + len(chunk) > end - start + 1:
                                raise RuntimeError('Server exceeded requested range')
                            check_disk(partial.parent)
                            f.write(chunk)
                            received += len(chunk)
                            progress('downloading', bytes=start + received, total_bytes=UBI_BYTES)
                    if received != end - start + 1:
                        raise RuntimeError(f'Truncated range: {received}')
                break
            except Exception:
                # Roll back only this failed range; previous verified ranges survive restart.
                if partial.exists():
                    with partial.open('r+b') as f:
                        f.truncate(start)
                if attempt == 4:
                    raise
                time.sleep(min(2**attempt, 15))
    partial.replace(archive)


def download_ubi_parallel(archive: Path, progress, workers: int = 8) -> None:
    """Resume independent verified parts; never interpret sparse-file length as progress."""
    if archive.exists():
        if archive.stat().st_size != UBI_BYTES:
            raise ValueError('Existing UBI archive has unexpected length')
        return
    parts = archive.parent / 'ubi_parts'
    parts.mkdir(parents=True, exist_ok=True)
    prefix = archive.with_suffix('.zip.part')
    base = prefix.stat().st_size if prefix.exists() else 0
    meta = parts / 'layout.json'
    if meta.exists():
        layout = json.loads(meta.read_text())
        if layout['prefix_bytes'] != base:
            raise ValueError('UBI prefix changed since part layout was frozen')
    else:
        layout = {'prefix_bytes': base, 'total_bytes': UBI_BYTES, 'part_bytes': 64 * 1024**2}
        write_json(meta, layout)
    ranges = [(s, min(s + layout['part_bytes'], UBI_BYTES) - 1)
              for s in range(base, UBI_BYTES, layout['part_bytes'])]
    completed = {s: (e-s+1 if (parts / f'{s}.part').exists() and (parts / f'{s}.part').stat().st_size == e-s+1 else 0)
                 for s, e in ranges}
    check_disk(parts, UBI_BYTES * 2 - base - sum(completed.values()))
    lock = threading.Lock()
    def one(pair):
        s, e = pair
        dest = parts / f'{s}.part'
        if completed[s]:
            return
        temp = parts / f'{s}.pending'
        for attempt in range(5):
            try:
                with requests.get(UBI_URL, headers={'Range': f'bytes={s}-{e}', 'Accept-Encoding': 'identity'},
                                  stream=True, timeout=(20, 120), verify=False, allow_redirects=False) as r:
                    if r.status_code != 206:
                        raise RuntimeError(f'UBI range returned HTTP {r.status_code}')
                    check_content_range(r.headers.get('Content-Range'), s, e, UBI_BYTES)
                    n = 0
                    with temp.open('wb') as f:
                        for b in r.iter_content(1024**2):
                            check_disk(parts)
                            n += len(b)
                            if n > e-s+1:
                                raise RuntimeError('Server exceeded range')
                            f.write(b)
                    if n != e-s+1:
                        raise RuntimeError('Incomplete UBI part')
                temp.replace(dest)
                with lock:
                    completed[s] = n
                    progress('downloading', bytes=base+sum(completed.values()), total_bytes=UBI_BYTES, workers=workers)
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2**attempt, 15))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in as_completed(pool.submit(one, pair) for pair in ranges):
            f.result()
    joined = archive.with_suffix('.zip.assembling')
    with joined.open('wb') as out:
        if base:
            with prefix.open('rb') as f:
                guarded_copy(f, out, archive.parent)
        for s, e in ranges:
            with (parts / f'{s}.part').open('rb') as f:
                guarded_copy(f, out, archive.parent)
    if joined.stat().st_size != UBI_BYTES:
        raise ValueError('Assembled archive length mismatch')
    joined.replace(archive)


def download_scfd(archive: Path, progress) -> None:
    if archive.exists():
        return
    check_disk(archive.parent, 100 * 1024**2)
    partial = archive.with_suffix('.zip.part')
    for attempt in range(5):
        try:
            with requests.get(SCFD_URL, stream=True, timeout=(20, 120)) as r:
                r.raise_for_status()
                n = 0
                with partial.open('wb') as f:
                    for chunk in r.iter_content(1024**2):
                        check_disk(archive.parent)
                        f.write(chunk)
                        n += len(chunk)
                        progress('downloading', bytes=n, total_bytes=int(r.headers.get('Content-Length', 0)))
            with zipfile.ZipFile(partial) as z:
                if z.testzip() is not None:
                    raise ValueError('SCFD archive CRC failure')
            partial.replace(archive)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 15))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['scfd', 'ubi'], required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status_path = OUT / f'download_{a.dataset}.json'
    last_log = [0.0]
    def progress(status, **extra):
        state = dict(dataset=a.dataset, status=status, pid=os.getpid(), started=started,
                     updated=time.time(), tls_exception=(UBI_URL if a.dataset == 'ubi' else None), **extra)
        write_json(status_path, state)
        if time.time() - last_log[0] > 20 or status != 'downloading':
            print(json.dumps(state), flush=True)
            last_log[0] = time.time()
    try:
        progress('starting')
        archive = DATA / ('UBI_FIGHTS.zip' if a.dataset == 'ubi' else f'scfd_{SCFD_COMMIT}.zip')
        if a.dataset == 'ubi':
            download_ubi_parallel(archive, progress, a.workers)
        else:
            download_scfd(archive, progress)
        progress('extracting', archive=str(archive), archive_bytes=archive.stat().st_size)
        files = safe_extract(archive, DATA / a.dataset)
        videos = [x for x in files if x['path'].lower().endswith('.mp4')]
        expected = 1000 if a.dataset == 'ubi' else 300
        if len(videos) != expected:
            raise RuntimeError(f'Expected {expected} videos, found {len(videos)}')
        if a.dataset == 'scfd' and sum(x['size'] for x in videos) != 91539597:
            raise RuntimeError('SCFD video byte count differs from pinned GitHub tree')
        receipt = dict(dataset=a.dataset, status='complete', url=UBI_URL if a.dataset == 'ubi' else SCFD_URL,
                       commit=SCFD_COMMIT if a.dataset == 'scfd' else None,
                       tls_exception=(dict(url=UBI_URL, reason='author certificate expired; explicit user authorization',
                                           redirects=False, global_verification_disabled=False) if a.dataset == 'ubi' else None),
                       archive=str(archive), archive_bytes=archive.stat().st_size,
                       archive_sha256=sha256(archive), files=files, completed=time.time())
        write_json(OUT / f'{a.dataset}_receipt.json', receipt)
        progress('complete', files=len(files), videos=len(videos), receipt=str(OUT / f'{a.dataset}_receipt.json'))
    except Exception as exc:
        progress('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
