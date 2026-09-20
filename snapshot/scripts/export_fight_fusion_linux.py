"""Export an auditable Linux continuation without changing the source experiment.

The copy receives explicit new provenance for path/platform adaptations. Model,
optimizer, scaler, RNG, cursor, split and selection semantics are preserved.
Rebuildable features, download archives, secrets and local environments are omitted.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path('results/fight_fusion_v1')
PRETRAIN_NAME = 'r3d_18-b3b3357e.pth'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def relative_source(value, root=ROOT):
    win = PureWindowsPath(value)
    if win.is_absolute():
        relative = win.relative_to(PureWindowsPath(str(root)))
        value = relative.as_posix()
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError('Source escapes the project: ' + value)
    return candidate.relative_to(root.resolve()).as_posix()


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(destination) != sha(source):
            raise ValueError('Conflicting export file: ' + str(destination))
        return
    shutil.copy2(source, destination)


def replace_once(path, before, after):
    # Keep all other source bytes, including original line endings, unchanged.
    data = path.read_bytes()
    old, new = before.encode(), after.encode()
    if data.count(old) != 1:
        raise ValueError('Unexpected source version: ' + str(path) + ': ' + before[:70])
    path.write_bytes(data.replace(old, new))


@contextmanager
def source_lock(path):
    with path.open('r+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def assert_equal_state(left, right, location='state'):
    import torch
    if type(left) is not type(right):
        raise ValueError('Checkpoint type changed: ' + location)
    if torch.is_tensor(left):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise ValueError('Checkpoint tensor changed: ' + location)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError('Checkpoint keys changed: ' + location)
        for key in left:
            assert_equal_state(left[key], right[key], location + '/' + str(key))
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise ValueError('Checkpoint sequence changed: ' + location)
        for i, (a, b) in enumerate(zip(left, right)):
            assert_equal_state(a, b, location + '/' + str(i))
    elif left != right:
        raise ValueError('Checkpoint value changed: ' + location)


def adapt_code(destination):
    run = destination / 'scripts/run_fight_fusion.py'
    data = run.read_text(encoding='utf-8')
    start = data.index('@contextmanager\ndef lock_queue():')
    end = data.index('\n\ndef save_status', start)
    replacement = '''@contextmanager
def lock_queue():
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'queue.lock').open('a+b') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
'''
    data = data[:start] + replacement + data[end:]
    if data.count('creationflags=subprocess.CREATE_NO_WINDOW') != 1:
        raise ValueError('Unexpected queue process creation')
    data = data.replace('creationflags=subprocess.CREATE_NO_WINDOW', 'creationflags=0')
    run.write_text(data, encoding='utf-8', newline='\n')
    # Keep model mathematics, optimization and sampling untouched.
    for name, before, after in [
        ('scripts/train_fight_fusion.py',
         'PRETRAIN = Path.home() / ".cache/torch/hub/checkpoints/r3d_18-b3b3357e.pth"',
         'PRETRAIN = ROOT / "pretrained/r3d_18-b3b3357e.pth"'),
        ('scripts/fight_fusion_features.py',
         "PRETRAIN = Path.home() / '.cache/torch/hub/checkpoints/r3d_18-b3b3357e.pth'",
         "PRETRAIN = ROOT / 'pretrained/r3d_18-b3b3357e.pth'"),
        ('backend/vision/fight_fusion.py',
         "KINETICS_PATH = Path.home() / '.cache/torch/hub/checkpoints/r3d_18-b3b3357e.pth'",
         "KINETICS_PATH = Path(__file__).resolve().parents[2] / 'pretrained/r3d_18-b3b3357e.pth'"),
        ('scripts/fight_fusion_features.py', 'str(p.relative_to(ROOT)): sha(p)',
         'p.relative_to(ROOT).as_posix(): sha(p)'),
    ]:
        replace_once(destination / name, before, after)


def migrate_checkpoint(source, target, old_config, new_config):
    import torch
    old = torch.load(source, map_location='cpu', weights_only=True)
    if old.get('config_signature') != signature(old_config):
        raise ValueError('Source checkpoint configuration is invalid: ' + str(source))
    # JSON serializes tuples as lists; checkpoints retain Python tuples for stages.
    # The trainer itself signs canonical JSON, so compare that same representation.
    if 'config' in old and signature(old['config']) != signature(old_config):
        raise ValueError('Source embedded config differs: ' + str(source))
    updated = dict(old, config_signature=signature(new_config))
    if 'config' in old:
        updated['config'] = new_config
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(updated, target)
    reloaded = torch.load(target, map_location='cpu', weights_only=True)
    if reloaded['config_signature'] != signature(new_config):
        raise ValueError('Migrated signature did not roundtrip')
    for key in old:
        if key not in ('config', 'config_signature'):
            assert_equal_state(old[key], reloaded[key], key)
    return dict(source_sha256=sha(source), exported_sha256=sha(target),
                state_verified_equal=True, metadata_changes=['config', 'config_signature'])


def create_bundle(destination, archive=True):
    import torch
    if str(torch.__version__) != '2.5.1+cu121':
        raise RuntimeError('Export using the original torch 2.5.1+cu121 environment')
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError('Use a new export directory: ' + str(destination))
    if shutil.disk_usage(destination.parent if destination.parent.exists() else ROOT).free < 28 * 1024**3:
        raise RuntimeError('Need at least 28 GiB free to stage and archive the bundle')
    source = ROOT / EXPERIMENT
    protocol = read(source / 'queue_protocol.json')
    manifest = read(source / 'data/manifest.json')
    old_manifest_sha = sha(source / 'data/manifest.json')
    old_seal = read(source / 'data/manifest.seal.json')
    if manifest.get('status') != 'sealed' or not manifest.get('training_allowed'):
        raise ValueError('Source data is not sealed')
    if protocol['manifest_sha256'] != old_manifest_sha or old_seal['manifest_sha256'] != old_manifest_sha:
        raise ValueError('Source data seal changed')
    for name, expected in protocol['code'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen source code changed: ' + name)
    if sha(ROOT / 'config/live_actions.json') != protocol['baseline_config_sha256']:
        raise ValueError('Frozen production baseline changed')
    if (source / 'fusion').exists() or (source / 'branches/skeleton_arm_selection.json').exists():
        raise ValueError('This exporter targets the current pre-fusion snapshot; later stages require a new migration audit')
    destination.mkdir(parents=True)
    original = destination / 'migration/original'
    records = dict(version=1, type='cross_platform_training_continuation', exported_at=time.time(),
        source_root=str(ROOT), source_manifest_sha256=old_manifest_sha,
        source_queue_protocol_sha256=sha(source / 'queue_protocol.json'),
        omitted=['rebuildable features', 'download ZIPs/parts', 'local virtual environments', '.env'],
        bitwise_reproducibility_claimed=False, checkpoints={}, path_map=[])
    # Source files only; never copy local secrets, environments or generated docs.
    for base in ['backend', 'scripts']:
        for path in (ROOT / base).rglob('*.py'):
            if '__pycache__' not in path.parts:
                copy_file(path, destination / path.relative_to(ROOT))
    for path in (ROOT / 'scripts/linux_fight').iterdir():
        if path.is_file():
            copy_file(path, destination / path.relative_to(ROOT))
    for path in (ROOT / 'third_party').glob('STGCNPP*'):
        if path.is_file():
            copy_file(path, destination / path.relative_to(ROOT))
    for path in (ROOT / 'tests').glob('test_fight_fusion*.py'):
        copy_file(path, destination / path.relative_to(ROOT))
    for name in ['tests/conftest.py', 'docs/FIGHT_FUSION_V1.md']:
        copy_file(ROOT / name, destination / name)
    for name in protocol['code']:
        copy_file(ROOT / name, original / name)
    copy_file(source / 'queue_protocol.json', original / EXPERIMENT / 'queue_protocol.json')
    for path in (source / 'data').iterdir():
        if path.suffix in ('.json', '.md'):
            copy_file(path, destination / EXPERIMENT / 'data' / path.name)
    for name in ['manifest.json', 'manifest.seal.json']:
        copy_file(source / 'data' / name, original / EXPERIMENT / 'data' / name)
    for name in ['models/yolov8n-pose.pt', 'models/stgcnpp/ntu60_xsub_hrnet_joint.pth']:
        copy_file(ROOT / name, destination / name)
    copy_file(Path.home() / '.cache/torch/hub/checkpoints' / PRETRAIN_NAME,
              destination / 'pretrained' / PRETRAIN_NAME)
    portable_manifest = copy.deepcopy(manifest)
    for index, row in enumerate(portable_manifest['rows']):
        for field, digest_field in [('path', 'sha256'), ('annotation_path', 'annotation_sha256')]:
            if not row.get(field):
                continue
            old_path = row[field]
            relative = relative_source(old_path)
            copy_file(ROOT / relative, destination / relative)
            if sha(destination / relative) != row[digest_field]:
                raise ValueError('Source content differs from sealed manifest: ' + relative)
            records['path_map'].append(dict(old=old_path, new=relative, sha256=row[digest_field]))
            row[field] = relative
        # Prove labels, splits, durations, frame annotations and IDs did not change.
        old_row = manifest['rows'][index]
        if {k: v for k, v in row.items() if k not in ('path', 'annotation_path')} != {
                k: v for k, v in old_row.items() if k not in ('path', 'annotation_path')}:
            raise ValueError('Source semantics changed')
        if (index + 1) % 500 == 0:
            print('COPIED_AND_HASH_VERIFIED_VIDEOS', index + 1, '/', len(manifest['rows']), flush=True)
    manifest_path = destination / EXPERIMENT / 'data/manifest.json'
    write(manifest_path, portable_manifest)
    portable_seal = dict(old_seal, manifest_sha256=sha(manifest_path),
        migration_source_manifest_sha256=old_manifest_sha, migration='Windows paths to portable relative paths; source bytes/splits/labels unchanged')
    write(manifest_path.with_suffix('.seal.json'), portable_seal)
    baseline = read(ROOT / 'config/live_actions.json')
    copy_file(ROOT / 'config/live_actions.json', original / 'config/live_actions.json')
    def relocate_config(value):
        if isinstance(value, dict):
            return {k: relocate_config(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate_config(v) for v in value]
        if isinstance(value, str) and value.lower().startswith(str(ROOT).lower() + '\\'):
            relative = relative_source(value)
            if not (ROOT / relative).is_file():
                raise FileNotFoundError('Baseline artifact missing: ' + value)
            copy_file(ROOT / relative, destination / relative)
            return relative
        return value
    baseline = relocate_config(baseline)
    baseline['runtime']['python'] = 'python'
    write(destination / 'config/live_actions.json', baseline)
    adapt_code(destination)
    command = "from scripts.fight_fusion_features import FeatureStore; s=FeatureStore(device='cpu',allow_build=False); print(s.signature); s.close()"
    result = subprocess.run([sys.executable, '-c', command], cwd=destination, capture_output=True, text=True, check=True)
    feature_signature = result.stdout.strip().splitlines()[-1]
    if len(feature_signature) != 64:
        raise ValueError('Could not compute portable feature signature')
    records['feature_signature'] = feature_signature
    for folder in sorted((source / 'branches').glob('*/seed*')):
        old_config = read(folder / 'config.json')
        for name, expected in old_config['code_hashes'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Branch code changed: ' + name)
        if old_config['manifest_sha256'] != old_manifest_sha:
            raise ValueError('Source branch manifest mismatch')
        if old_config['manifest_seal_sha256'] != sha(source / 'data/manifest.seal.json'):
            raise ValueError('Source branch seal mismatch')
        config = dict(old_config, manifest_sha256=sha(manifest_path),
            manifest_seal_sha256=sha(manifest_path.with_suffix('.seal.json')),
            feature_signature=feature_signature,
            code_hashes={name: sha(destination / name) for name in old_config['code_hashes']})
        target = destination / folder.relative_to(ROOT)
        for path in folder.glob('*.json'):
            copy_file(path, original / path.relative_to(ROOT))
            if path.name not in ('config.json', 'selection.json'):
                value = read(path)
                if isinstance(value, dict) and 'config_signature' in value:
                    if value['config_signature'] != signature(old_config):
                        raise ValueError('Source record config signature mismatch')
                    value['config_signature'] = signature(config)
                write(target / path.name, value)
        write(target / 'config.json', config)
        completed = (folder / 'selection.json').exists()
        checkpoints = [folder / 'selected_best.pt'] if completed else sorted(folder.glob('*.pt'))
        for path in checkpoints:
            records['checkpoints'][path.relative_to(ROOT).as_posix()] = migrate_checkpoint(path, target / path.name, old_config, config)
        if completed:
            selected = read(folder / 'selection.json')
            if selected['config_signature'] != signature(old_config) or selected['sha256'] != sha(folder / 'selected_best.pt'):
                raise ValueError('Source selection checkpoint mismatch')
            selected.update(config_signature=signature(config), sha256=sha(target / 'selected_best.pt'),
                            checkpoint=(folder.relative_to(ROOT) / 'selected_best.pt').as_posix())
            write(target / 'selection.json', selected)
        else:
            resume = torch.load(target / 'resume.pt', map_location='cpu', weights_only=True)['runner_state']
            records['resume'] = {k: resume[k] for k in ('epochs_total', 'cursor', 'optimizer_steps', 'stage_index', 'stage_epoch', 'patience')}
            write(target / 'progress.json', dict(phase='paused_for_linux_migration', branch=config['branch'],
                seed=config['seed'], epoch=resume['epochs_total'] + 1, completed=resume['cursor'],
                total=len(config['train_sample_ids']), last_atomic_checkpoint=True))
    # Retain original Windows smoke evidence byte-for-byte, separately from the
    # required fresh Linux GPU smoke acceptance.
    for branch in ('global', 'roi', 'skeleton_random', 'skeleton_ntu'):
        folder = source / 'smoke_outputs_v4/smoke' / branch / 'seed42'
        selected = read(folder / 'selection.json')
        if selected['sha256'] != sha(folder / 'selected_best.pt') or selected['status'] != 'complete':
            raise ValueError('Original GPU smoke is invalid')
        for name in ['selection.json', 'selected_best.pt']:
            copy_file(folder / name, destination / folder.relative_to(ROOT) / name)
    records['changed_code'] = {name: dict(source_sha256=sha(ROOT / name), portable_sha256=sha(destination / name))
        for name in protocol['code'] if sha(ROOT / name) != sha(destination / name)}
    records['source_files_unchanged_verified'] = True
    for name, expected in protocol['code'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Source changed during export')
    exported_protocol = dict(protocol, manifest_sha256=sha(manifest_path),
        code={name: sha(destination / name) for name in protocol['code']},
        baseline_config_sha256=sha(destination / 'config/live_actions.json'),
        migration_source_protocol_sha256=records['source_queue_protocol_sha256'])
    write(destination / EXPERIMENT / 'queue_protocol.json', exported_protocol)
    write(destination / 'migration/record.json', records)
    readme = ROOT / 'docs/FIGHT_FUSION_LINUX.md'
    copy_file(readme, destination / 'README_LINUX.md')
    files = []
    for path in sorted(destination.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        relative = path.relative_to(destination).as_posix()
        if relative.startswith('datasets/fight_fusion_v1/features/'):
            continue
        mutable = relative.startswith('results/fight_fusion_v1/branches/')
        files.append(dict(path=relative, bytes=path.stat().st_size, sha256=sha(path), immutable=not mutable))
    write(destination / 'migration/inventory.json', dict(version=1, files=files))
    if archive:
        target = destination.with_suffix('.tar')
        if target.exists():
            raise FileExistsError(target)
        print('ARCHIVING', target, flush=True)
        archive_paths = [entry['path'] for entry in files] + ['migration/inventory.json']
        with tarfile.open(target, mode='x', dereference=True) as tar:
            for name in archive_paths:
                path = destination / name
                info = tar.gettarinfo(str(path), arcname='fight_fusion_linux/' + name)
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.mode = 0o755 if path.suffix == '.sh' else 0o644
                with path.open('rb') as handle:
                    tar.addfile(info, handle)
        expected = {entry['path']: entry['sha256'] for entry in files}
        expected['migration/inventory.json'] = sha(destination / 'migration/inventory.json')
        seen = set()
        with tarfile.open(target, 'r') as tar:
            for member in tar:
                name = member.name.removeprefix('fight_fusion_linux/')
                if name not in expected or name in seen or not member.isfile():
                    raise ValueError('Unexpected archive entry')
                with tar.extractfile(member) as handle:
                    if hashlib.file_digest(handle, 'sha256').hexdigest() != expected[name]:
                        raise ValueError('Archive checksum mismatch: ' + name)
                seen.add(name)
        if seen != expected.keys():
            raise ValueError('Archive is incomplete')
        checksum = sha(target)
        target.with_suffix('.tar.sha256').write_text(checksum + '  ' + target.name + '\n', encoding='ascii')
        receipt = dict(archive=str(target), bytes=target.stat().st_size, gib=target.stat().st_size / 1024**3,
            sha256=checksum, archived_files=len(seen), all_archive_members_hash_verified=True,
            copied_source_videos=len(manifest['rows']), resume=records.get('resume'),
            original_experiment_unchanged=True, target_linux_gpu_validated=False)
        write(target.with_suffix('.receipt.json'), receipt)
        print(json.dumps(receipt, indent=2), flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'exports' / ('fight_fusion_linux_' + time.strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--no-archive', action='store_true')
    args = parser.parse_args()
    with ExitStack() as stack:
        for name in ('queue.lock', 'branch_trainer.lock'):
            stack.enter_context(source_lock(ROOT / EXPERIMENT / name))
        create_bundle(args.output, archive=not args.no_archive)


if __name__ == '__main__':
    main()
