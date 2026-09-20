"""Validate a Linux continuation and run isolated real-video GPU smoke checks."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.linux_fight.verify import read, sha, verify

EXPECTED = {'torch': '2.5.1+cu121', 'torchvision': '0.20.1+cu121',
            'numpy': '1.26.4', 'opencv-python': '4.9.0.80', 'ultralytics': '8.2.0'}


def validate_configs(torch):
    from scripts import train_fight_fusion as trainer
    from scripts.fight_fusion_features import FeatureStore
    from scripts.fit_fight_fusion import verify_data_seal
    from backend.vision import fight_fusion as models
    experiment = ROOT / 'results/fight_fusion_v1'
    manifest_path = experiment / 'data/manifest.json'
    manifest = verify_data_seal(manifest_path)
    trainer.validate_manifest_seal(manifest, manifest_path)
    store = FeatureStore(manifest, device='cpu', allow_build=False)
    try:
        checked = []
        for folder in sorted((experiment / 'branches').glob('*/seed*')):
            if not (folder / 'config.json').is_file():
                continue
            config = read(folder / 'config.json')
            signature = trainer.signature(config)
            args = SimpleNamespace(branch=config['branch'], seed=config['seed'],
                                   smoke=False, max_epochs=None, device='cuda')
            class ConfigChecked(Exception):
                pass
            def stop_before_model(*a, **kw):
                raise ConfigChecked()
            originals = models.make_rgb_model, models.FusionSkeletonModel, trainer.seed_everything
            # Use the real trainer's config-building/validation path, in a temporary
            # directory; do not modify the experiment or allocate a CUDA model.
            try:
                models.make_rgb_model = models.FusionSkeletonModel = stop_before_model
                trainer.seed_everything = lambda seed: None
                with tempfile.TemporaryDirectory(prefix='fight-config-') as tmp:
                    target = Path(tmp)
                    shutil.copyfile(folder / 'config.json', target / 'config.json')
                    try:
                        trainer._train_with_store(args, manifest, manifest_path, target, store)
                    except ConfigChecked:
                        pass
                    else:
                        raise RuntimeError('Unexpected config probe control flow')
            finally:
                models.make_rgb_model, models.FusionSkeletonModel, trainer.seed_everything = originals
            for path in folder.glob('*.pt'):
                payload = torch.load(path, map_location='cpu', weights_only=True)
                if payload.get('config_signature') != signature:
                    raise ValueError('Checkpoint config mismatch: ' + str(path))
                if 'config' in payload and payload['config'] != config:
                    raise ValueError('Checkpoint embedded config mismatch: ' + str(path))
                del payload
            if (folder / 'selection.json').is_file():
                selected = read(folder / 'selection.json')
                if selected['sha256'] != sha(folder / 'selected_best.pt') or selected['config_signature'] != signature:
                    raise ValueError('Selected checkpoint provenance mismatch')
            checked.append(dict(branch=config['branch'], seed=config['seed'], signature=signature))
        return checked, store.signature
    finally:
        store.close()


def gpu_identity(torch):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU is required; set CUDA_VISIBLE_DEVICES')
    # Actually execute a CUDA kernel and cuDNN operation. A driver/device listing
    # alone cannot demonstrate that this PyTorch build supports the target GPU.
    x = torch.ones((1, 1, 4, 8, 8), device='cuda', requires_grad=True)
    convolution = torch.nn.Conv3d(1, 2, 3, padding=1).cuda()
    convolution(x).sum().backward()
    torch.cuda.synchronize()
    result = dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                  torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  platform=platform.platform(), python=platform.python_version(),
                  libraries={k: importlib.metadata.version(k) for k in EXPECTED})
    del x, convolution
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-smoke', action='store_true', help='Run all four branches on isolated real training/validation clips')
    parser.add_argument('--cpu-check', action='store_true', help='Offline package/config check only; never authorizes formal training')
    args = parser.parse_args()
    if args.cpu_check and args.gpu_smoke:
        parser.error('Choose one validation mode')
    os.chdir(ROOT)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.update(YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false')
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Python 3.12 is required for this continuation')
    if not args.cpu_check and sys.platform != 'linux':
        raise RuntimeError('Formal continuation and GPU acceptance require Linux')
    for name, expected in EXPECTED.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f'{name}: expected {expected}, got {actual}; do not silently upgrade a continuation')
    print(json.dumps(verify()), flush=True)
    import torch
    torch.set_num_threads(4)
    branches, feature_signature = validate_configs(torch)
    print(json.dumps(dict(configs_validated=branches, feature_signature=feature_signature)), flush=True)
    if args.cpu_check:
        print('CPU_PACKAGE_CHECK_PASSED; target Linux GPU execution still required', flush=True)
        return
    from scripts.linux_fight.guard import memory_available, PressureMonitor, gpu_snapshot
    reason = PressureMonitor().check(memory_available(), gpu_snapshot()['free_mib'], starting=True)
    if reason:
        raise RuntimeError(reason)
    # Existing cache may reduce future growth; preserve the protocol's 100 GiB reserve.
    cache = ROOT / 'datasets/fight_fusion_v1/features'
    cache_bytes = sum(p.stat().st_size for p in cache.rglob('*.pt'))
    required = (100 + 128 + 8) * 1024**3 - min(cache_bytes, 128 * 1024**3)
    if shutil.disk_usage(ROOT).free < required:
        raise RuntimeError(f'Need {required / 1024**3:.1f} GiB free here for cache growth, outputs and 100 GiB reserve')
    identity = gpu_identity(torch)
    acceptance_path = ROOT / 'migration/linux_gpu_acceptance.json'
    bundle_signature = sha(ROOT / 'migration/inventory.json')
    if args.gpu_smoke:
        if acceptance_path.exists():
            acceptance = read(acceptance_path)
            if acceptance['identity'] == identity and acceptance['bundle_signature'] == bundle_signature:
                print('Existing Linux GPU acceptance matches this bundle and environment', flush=True)
            else:
                raise RuntimeError('Existing acceptance differs; retain it and investigate environment change')
        else:
            # Timestamped output prevents mixing smoke runs between different GPUs.
            output = ROOT / 'results/linux_gpu_smoke' / time.strftime('%Y%m%d_%H%M%S')
            output.mkdir(parents=True, exist_ok=False)
            selections = []
            for branch in ('global', 'roi', 'skeleton_random', 'skeleton_ntu'):
                command = [sys.executable, '-u', 'scripts/train_fight_fusion.py', '--branch', branch,
                    '--seed', '42', '--smoke', '--max-epochs', '1', '--output-root', str(output)]
                with (output / (branch + '.log')).open('wb') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                path = output / 'smoke' / branch / 'seed42/selection.json'
                record = read(path)
                if record['status'] != 'complete' or not record['smoke'] or sha(path.with_name('selected_best.pt')) != record['sha256']:
                    raise ValueError('GPU smoke checkpoint verification failed')
                selections.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                                       checkpoint_sha256=record['sha256']))
                print('LINUX_GPU_SMOKE_PASSED', branch, flush=True)
            acceptance = dict(identity=identity, bundle_signature=bundle_signature,
                completed=time.time(), selections=selections, formal_training_started=False,
                limitation='Cross-platform and cross-GPU continuation is not bitwise reproducible')
            acceptance_path.write_text(json.dumps(acceptance, indent=2), encoding='utf-8')
    if not acceptance_path.is_file():
        raise RuntimeError('First run: python scripts/linux_fight/preflight.py --gpu-smoke')
    acceptance = read(acceptance_path)
    if acceptance['identity'] != identity or acceptance['bundle_signature'] != bundle_signature:
        raise ValueError('Linux GPU acceptance no longer matches current environment/bundle')
    for entry in acceptance['selections']:
        path = ROOT / entry['path']
        if sha(path) != entry['sha256'] or sha(path.with_name('selected_best.pt')) != entry['checkpoint_sha256']:
            raise ValueError('Linux GPU smoke artifacts changed')
    print('LINUX_CONTINUATION_READY', json.dumps(identity), flush=True)


if __name__ == '__main__':
    main()
