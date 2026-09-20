"""Preserve verified active A/B checkpoint sets without modifying training files."""
from pathlib import Path
import ctypes
from ctypes import wintypes
import datetime
import hashlib
import json
import msvcrt
import os
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import stgcnpp_ab as ab


def durable_json(path, value):
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def shared_copy(source, destination):
    # SHARE_DELETE is required: ordinary readers can block checkpoint replacement.
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
        wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
        wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    handle = kernel.CreateFileW(str(source), 0x80000000, 7, None, 3, 0x08000080, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, 'rb') as reader, destination.open('xb') as writer:
        for chunk in iter(lambda: reader.read(1024 * 1024), b''):
            writer.write(chunk)
            digest.update(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    return digest.hexdigest()


def main():
    ab.torch.set_num_threads(1)
    for attempt in range(3):
        try:
            queue = ab.read(ab.OUT / 'queue_status.json')
            break
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(0.2)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination = ab.OUT / 'recoveries' / 'active_snapshots' / stamp
    destination.mkdir(parents=True, exist_ok=False)
    receipt = dict(at=datetime.datetime.now().isoformat(), models=[],
                   training_files_modified=False, copy_sharing='READ|WRITE|DELETE')
    for item in queue.get('active_models', []):
        name, arm, seed = item['dataset'], item['arm'], item['seed']
        source = ab.OUT / name / arm / f'seed_{seed}'
        target = destination / name / arm / f'seed_{seed}'
        target.mkdir(parents=True)
        record = dict(dataset=name, arm=arm, seed=seed, status='incomplete')
        try:
            hashes = {}
            for filename in ('resume.pt', 'best.pt', 'best_validation_predictions.json',
                             'config.json', 'initialization.json'):
                hashes[filename] = shared_copy(source / filename, target / filename)
            for filename in ('resume.pt', 'best.pt'):
                with zipfile.ZipFile(target / filename) as archive:
                    assert archive.testzip() is None, 'Archive CRC failed'
            saved = ab.torch.load(target / 'resume.pt', map_location='cpu', weights_only=True)
            best = ab.torch.load(target / 'best.pt', map_location='cpu', weights_only=True)
            config = ab.model_config(name, arm, seed)
            signature = ab.digest(config)
            assert ab.read(target / 'config.json') == config
            assert saved['config_signature'] == best['config_signature'] == signature
            assert saved['optimizer']['state'] and saved['scaler'] and saved['rng']
            state = saved['runner_state']
            history = next(h for h in state['history'] if h['epoch'] == best['epoch'])
            assert history['threshold'] == best['threshold']
            assert history['validation'] == best['validation']
            assert list(ab.selection_key(best['validation'])) == state['best_key']
            predictions = ab.read(target / 'best_validation_predictions.json')
            assert predictions['threshold'] == best['threshold']
            assert all(ab.torch.isfinite(t).all().item() for t in saved['model'].values()
                       if isinstance(t, ab.torch.Tensor) and t.is_floating_point())
            durable_json(target / 'run_record.json',
                         dict(status='training', config_signature=signature, **state))
            record.update(status='verified', sha256=hashes, epoch=state['epoch'] + 1,
                          cursor=state['cursor'], phase=state['phase'],
                          optimizer_steps=state['optimizer_steps'],
                          epochs_finished=len(state['history']), best_epoch=best['epoch'])
            del saved, best
        except Exception as exc:
            # An epoch boundary can change the best checkpoint while it is copied.
            # Preserve evidence, but never advertise an inconsistent set as usable.
            record['error'] = f'{type(exc).__name__}: {exc}'
        receipt['models'].append(record)
    receipt['status'] = ('verified' if receipt['models'] and
                         all(m['status'] == 'verified' for m in receipt['models'])
                         else 'incomplete_or_no_active_models')
    durable_json(destination / 'snapshot.json', receipt)
    print(json.dumps(dict(path=str(destination / 'snapshot.json'), **receipt), ensure_ascii=True))


if __name__ == '__main__':
    main()
