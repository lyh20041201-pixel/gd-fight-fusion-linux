"""Linux-only memory watchdog for the exported queue; no training imports."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/fight_fusion_v1'
GIB = 1024 ** 3


def memory_available(proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')):
    info = {line.split(':', 1)[0]: int(line.split()[1]) * 1024
            for line in (proc / 'meminfo').read_text().splitlines() if ':' in line}
    available = info['MemAvailable']
    checks = [(cgroup, 'memory.max', 'memory.current'),
              (cgroup / 'memory', 'memory.limit_in_bytes', 'memory.usage_in_bytes')]
    for line in (proc / 'self/cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        base = cgroup if controllers == '' else cgroup / 'memory' if 'memory' in controllers.split(',') else None
        if base is None:
            continue
        current = (base / relative.lstrip('/')).resolve()
        if not current.is_relative_to(base.resolve()):
            continue
        while current.is_relative_to(base.resolve()):
            names = ('memory.max', 'memory.current') if controllers == '' else ('memory.limit_in_bytes', 'memory.usage_in_bytes')
            checks.append((current, *names))
            if current == base.resolve():
                break
            current = current.parent
    for folder, limit_name, used_name in checks:
        if (folder / limit_name).is_file() and (folder / used_name).is_file():
            value = (folder / limit_name).read_text().strip()
            if value != 'max':
                available = min(available, max(0, int(value) - int((folder / used_name).read_text())))
    return available / GIB


def gpu_snapshot():
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    if not selected or ',' in selected:
        raise ValueError('Select exactly one physical GPU with CUDA_VISIBLE_DEVICES')
    result = subprocess.run(['nvidia-smi', '--id=' + selected,
        '--query-gpu=memory.free,memory.used,memory.total', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=10)
    values = [float(x.strip()) for x in result.stdout.strip().split(',')]
    if len(values) != 3:
        raise ValueError('Expected exactly one NVIDIA GPU')
    return dict(zip(('free_mib', 'used_mib', 'total_mib'), values))


class PressureMonitor:
    def __init__(self):
        self.memory_low = self.gpu_low = 0

    def check(self, available, gpu_free, starting=False):
        if starting and available < 8:
            return 'At least 8 GiB of available host/container RAM is required to start'
        if available < 4:
            return 'Available host/container RAM is below 4 GiB'
        self.memory_low = self.memory_low + 1 if available < 6 else 0
        self.gpu_low = self.gpu_low + 1 if gpu_free < 256 else 0
        if starting and self.gpu_low:
            return 'GPU free memory is below 256 MiB'
        if self.memory_low >= 3:
            return 'Available RAM was below 6 GiB for three consecutive samples'
        if self.gpu_low >= 3:
            return 'GPU free memory was below 256 MiB for three consecutive samples'
        return None


@contextmanager
def exclusive_lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def write_status(value):
    path = OUT / 'linux_guard_status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def stop_group(process):
    # This group was created specifically for this invocation, never an unrelated process.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def run():
    if sys.platform != 'linux':
        raise RuntimeError('This entry point requires Linux')
    os.chdir(ROOT)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    # Run all validation before starting any formal trainer.
    subprocess.run([sys.executable, str(ROOT / 'scripts/linux_fight/preflight.py')], check=True)
    with exclusive_lock(OUT / 'linux_guard.lock'):
        monitor = PressureMonitor()
        gpu = gpu_snapshot()
        reason = monitor.check(memory_available(), gpu['free_mib'], starting=True)
        if reason:
            raise RuntimeError(reason)
        logpath = OUT / 'logs' / ('linux_queue_' + time.strftime('%Y%m%d_%H%M%S') + '.log')
        logpath.parent.mkdir(parents=True, exist_ok=True)
        def interrupted(signum, frame):
            raise InterruptedError('Received signal ' + str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        with logpath.open('ab') as log:
            process = subprocess.Popen([sys.executable, '-u', 'scripts/run_fight_fusion.py'],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True)
            print('QUEUE_LOG', logpath, flush=True)
            try:
                while process.poll() is None:
                    available, gpu = memory_available(), gpu_snapshot()
                    reason = monitor.check(available, gpu['free_mib'])
                    if reason:
                        raise RuntimeError(reason)
                    write_status(dict(status='running', pid=os.getpid(), queue_pid=process.pid,
                        updated=time.time(), available_gib=available, gpu=gpu, log=str(logpath)))
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                if process.returncode:
                    raise RuntimeError('Queue failed, exit=' + str(process.returncode) + '; see ' + str(logpath))
            except BaseException as exc:
                stop_group(process)
                write_status(dict(status='attention_required', reason=str(exc), updated=time.time(),
                    log=str(logpath), automatic_restart=False, last_atomic_checkpoint_preserved=True))
                raise
            finally:
                if process.poll() is None:
                    stop_group(process)
            write_status(dict(status='complete', updated=time.time(), log=str(logpath)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        print(json.dumps(dict(available_gib=memory_available(), gpu=gpu_snapshot()), indent=2))
    else:
        run()
