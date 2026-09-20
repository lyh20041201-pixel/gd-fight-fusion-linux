#!/usr/bin/env python3
"""Fetch the sealed GitHub Release and continue training in one tmux session."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def manifest():
    value = read(ROOT / 'release-manifest.json')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', value['archive']['name']):
        raise ValueError('Invalid archive name')
    if not re.fullmatch(r'[0-9a-f]{64}', value['archive']['sha256']):
        raise ValueError('Invalid archive hash')
    names = set()
    for entry in value['parts']:
        name = entry['name']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in names:
            raise ValueError('Invalid or duplicate release asset name')
        if not re.fullmatch(r'[0-9a-f]{64}', entry['sha256']) or not 0 < entry['bytes'] < 2 * GIB:
            raise ValueError('Invalid release asset size/hash')
        names.add(name)
    if sum(p['bytes'] for p in value['parts']) != value['archive']['bytes']:
        raise ValueError('Part sizes do not match archive size')
    return value


def checked_file(path, entry):
    return path.is_file() and path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']


def asset_url(specification, name):
    repository = specification['repository']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository):
        raise ValueError('Invalid repository name')
    return (f'https://github.com/{repository}/releases/download/'
            f"{quote(specification['tag'], safe='')}/{quote(name, safe='')}")


def download_asset(url, destination, expected):
    """Stream a public asset without credentials; expose only verified complete files."""
    if checked_file(destination, expected):
        return
    temporary = destination.with_name(destination.name + '.downloading')
    request = Request(url, headers={'User-Agent': 'gd-fight-linux-bootstrap/1',
                                    'Accept': 'application/octet-stream'})
    received = 0
    digest = hashlib.sha256()
    last_report = time.monotonic()
    try:
        with urlopen(request, timeout=120) as response, temporary.open('wb') as output:
            while block := response.read(4 * 1024**2):
                received += len(block)
                if received > expected['bytes']:
                    raise ValueError('Downloaded asset exceeds expected size: ' + destination.name)
                output.write(block)
                digest.update(block)
                if time.monotonic() - last_report >= 30:
                    print(f'DOWNLOAD {destination.name}: {received}/{expected["bytes"]} bytes', flush=True)
                    last_report = time.monotonic()
        if received != expected['bytes'] or digest.hexdigest() != expected['sha256']:
            raise ValueError('Downloaded asset size/hash mismatch: ' + destination.name)
        temporary.replace(destination)
        print('VERIFIED', destination.name, flush=True)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run(command, **kwargs):
    print('RUN', shlex.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def python312():
    candidates = [os.environ.get('PYTHON_BIN'), shutil.which('python3.12'), sys.executable]
    for candidate in dict.fromkeys(p for p in candidates if p):
        result = subprocess.run([candidate, '-c', 'import sys;raise SystemExit(sys.version_info[:2] != (3,12))'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            return candidate
    raise RuntimeError('Python 3.12 is missing; provision it before starting this continuation')


def status(workdir):
    result = {}
    for name, path in [('controller', workdir / 'controller-status.json'),
                       ('guard', workdir / 'fight_fusion_linux/results/fight_fusion_v1/linux_guard_status.json'),
                       ('queue', workdir / 'fight_fusion_linux/results/fight_fusion_v1/queue_status.json')]:
        if path.is_file():
            result[name] = read(path)
    branch_root = workdir / 'fight_fusion_linux/results/fight_fusion_v1/branches'
    result['branches'] = {p.parent.parent.name + '/' + p.parent.name: read(p)
                          for p in branch_root.glob('*/seed*/progress.json')}
    return result


def assemble(downloads, specification):
    destination = downloads / specification['archive']['name']
    if checked_file(destination, specification['archive']):
        return destination
    temporary = destination.with_suffix('.assembling')
    with temporary.open('wb') as output:
        for part in specification['parts']:
            source = downloads / part['name']
            if not checked_file(source, part):
                raise ValueError('Bad or missing part: ' + part['name'])
            with source.open('rb') as handle:
                shutil.copyfileobj(handle, output, length=4 * 1024**2)
    if not checked_file(temporary, specification['archive']):
        raise ValueError('Assembled archive hash mismatch')
    temporary.replace(destination)
    return destination


def extract(archive, workdir):
    # Only ordinary files/directories below this exact top-level directory.
    with tarfile.open(archive, 'r') as handle:
        members = handle.getmembers()
        for member in members:
            relative = Path(member.name)
            if (not relative.parts or relative.parts[0] != 'fight_fusion_linux' or
                '..' in relative.parts or relative.is_absolute() or
                not (member.isfile() or member.isdir())):
                raise ValueError('Unexpected archive member: ' + member.name)
        temporary = workdir / '.extracting'
        temporary.mkdir(parents=True, exist_ok=True)
        handle.extractall(temporary, members=members, filter='data')
    target = workdir / 'fight_fusion_linux'
    if target.exists():
        raise FileExistsError('Will not overwrite a training workspace: ' + str(target))
    (temporary / 'fight_fusion_linux').rename(target)
    return target


def prerequisites(workdir, require_tmux=True):
    if sys.platform != 'linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise RuntimeError('This training continuation requires Linux x86_64')
    for executable in ('nvidia-smi', 'bash', *(['tmux'] if require_tmux else [])):
        if not shutil.which(executable):
            raise RuntimeError('Required command is missing: ' + executable)
    python = python312()
    run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'])
    workdir.mkdir(parents=True, exist_ok=True)
    if not (workdir / 'fight_fusion_linux').exists() and shutil.disk_usage(workdir).free < 285 * GIB:
        raise RuntimeError('Fresh download/extraction/environment/cache needs at least 285 GiB free in the selected work directory')
    return python


def worker(args, specification):
    import fcntl
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    with (workdir / 'controller.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (workdir / 'controller.log').open('a', encoding='utf-8', buffering=1) as log:
            sys.stdout = sys.stderr = log
            # Child stdout must use this same log, not the previous tmux descriptor.
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            def phase(name, **extra):
                data = dict(phase=name, updated=time.time(), pid=os.getpid(), gpu=args.gpu,
                            repository=specification['repository'], release=specification['tag'], **extra)
                write(workdir / 'controller-status.json', data)
                print(json.dumps(data, ensure_ascii=False), flush=True)
            try:
                phase('checking_prerequisites')
                python = prerequisites(workdir, require_tmux=False)
                os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHON_BIN=python,
                                  CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1')
                marker = workdir / 'bundle-ready.json'
                folder = workdir / 'fight_fusion_linux'
                if marker.exists():
                    if read(marker)['archive_sha256'] != specification['archive']['sha256'] or not folder.is_dir():
                        raise ValueError('Existing work directory belongs to a different bundle or is incomplete')
                else:
                    downloads = workdir / 'downloads'
                    downloads.mkdir(exist_ok=True)
                    # Public Release URLs require neither gh nor a GitHub account/token.
                    for index, part in enumerate(specification['parts']):
                        phase('downloading', completed=index, total=len(specification['parts']), asset=part['name'])
                        path = downloads / part['name']
                        download_asset(asset_url(specification, part['name']), path, part)
                    phase('assembling_verified_parts')
                    archive = assemble(downloads, specification)
                    phase('extracting')
                    # If extraction completed before an interruption, verify the existing
                    # directory rather than overwriting its potentially newer checkpoints.
                    if not folder.exists():
                        folder = extract(archive, workdir)
                    run([python, 'scripts/linux_fight/verify.py', '--full'], cwd=folder)
                    write(marker, dict(archive_sha256=specification['archive']['sha256'], verified_at=time.time()))
                environment_marker = workdir / 'environment-ready.json'
                if not environment_marker.exists():
                    phase('installing_environment')
                    run(['bash', 'scripts/linux_fight/setup.sh'], cwd=folder)
                    write(environment_marker, dict(archive_sha256=specification['archive']['sha256'], installed_at=time.time()))
                elif read(environment_marker)['archive_sha256'] != specification['archive']['sha256']:
                    raise ValueError('Environment marker belongs to a different bundle')
                training_python = folder / '.venv-linux/bin/python'
                phase('linux_gpu_smoke')
                run([training_python, 'scripts/linux_fight/preflight.py', '--gpu-smoke'], cwd=folder)
                phase('starting_training')
                run(['bash', 'scripts/linux_fight/start.sh'], cwd=folder)
                final = read(folder / 'results/fight_fusion_v1/queue_status.json')
                if final.get('status') != 'complete':
                    raise RuntimeError('Queue exited without a completion record')
                phase('complete', report=str(folder / 'results/fight_fusion_v1/REPORT.md'))
            except BaseException as error:
                phase('attention_required', error=str(error), automatically_restarted=False)
                raise
            finally:
                sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--start', action='store_true')
    mode.add_argument('--status', action='store_true')
    mode.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--workdir', type=Path, default=ROOT / '.training')
    parser.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    parser.add_argument('--session', default='gd-fight-train')
    args = parser.parse_args()
    args.workdir = args.workdir.expanduser().resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.session) or not args.gpu or ',' in args.gpu:
        parser.error('Use a simple tmux session name and exactly one CUDA GPU')
    if args.status:
        print(json.dumps(status(args.workdir), ensure_ascii=False, indent=2))
        return
    specification = manifest()
    if args.worker:
        worker(args, specification)
        return
    python = prerequisites(args.workdir)
    if subprocess.run(['tmux', 'has-session', '-t', '=' + args.session],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        print('Existing tmux session retained:', args.session)
        print(json.dumps(status(args.workdir), ensure_ascii=False, indent=2))
        return
    command = [python, str(Path(__file__).resolve()), '--worker', '--workdir', str(args.workdir), '--gpu', args.gpu]
    run(['tmux', 'new-session', '-d', '-s', args.session, '-c', ROOT, shlex.join(command)])
    print('Controller launched in tmux:', args.session)
    print('Log:', args.workdir / 'controller.log')
    print('Run --status and verify real training progress before declaring training started.')


if __name__ == '__main__':
    main()
