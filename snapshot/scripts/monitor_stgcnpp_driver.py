"""Observe the local A/B queue and stop it on a sustained NVIDIA error burst.

This operational guard does not import or change training code or model state.
"""
from pathlib import Path
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/video_events/skeleton_stgcnpp_ab'
POWERSHELL = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.guard_tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def events():
    script = r"""
$taskEnd = Get-Date
try {
  $taskRows = @(Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='nvlddmkm';StartTime=$taskEnd.AddSeconds(-60);EndTime=$taskEnd} -MaxEvents 20000 -ErrorAction Stop)
} catch {
  if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') { $taskRows = @() } else { throw }
}
[pscustomobject]@{at=$taskEnd.ToString('o');window_seconds=60;count=$taskRows.Count;capped=($taskRows.Count -eq 20000);latest=@($taskRows | Select-Object -First 3 @{n='record_id';e={$_.RecordId}},Id,@{n='time';e={$_.TimeCreated.ToString('o')}})} | ConvertTo-Json -Depth 4 -Compress
"""
    result = subprocess.run([str(POWERSHELL), '-NoProfile', '-NonInteractive', '-Command', script],
                            capture_output=True, text=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])
    return json.loads(result.stdout)


def stop_queue(queue, folder, observation):
    # Capture identities before touching anything; only this queue's descendants.
    processes = queue.children(recursive=True) + [queue]
    identities = [(p, p.create_time()) for p in processes]
    receipt = dict(at=dt.datetime.now().isoformat(), reason='NVIDIA error burst',
                   observation=observation, processes=[], snapshot=None,
                   threshold='>=500 events/min once or >=100 events/min for two consecutive checks')
    suspended = []
    try:
        # Freeze the queue first so it cannot create another worker during capture.
        for process, created in reversed(identities):
            try:
                if process.create_time() != created:
                    continue
                process.suspend()
                suspended.append(process)
                receipt['processes'].append(dict(pid=process.pid, created=created, command=process.cmdline()))
            except psutil.NoSuchProcess:
                pass
        try:
            result = subprocess.run([sys.executable, 'scripts/snapshot_stgcnpp_active.py'],
                                    cwd=ROOT, capture_output=True, text=True, timeout=45,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            receipt['snapshot'] = json.loads(result.stdout.strip()) if result.returncode == 0 else dict(error=result.stderr[-2000:])
        except Exception as exc:
            receipt['snapshot'] = dict(error=str(exc), fallback='Preserved pre-run verified checkpoints remain available')
        try:
            receipt['queue_before_stop'] = json.loads((OUT / 'queue_status.json').read_text(encoding='utf-8'))
        except Exception as exc:
            receipt['queue_read_error'] = str(exc)
        save(folder / 'driver_guard_stop.json', receipt)
        failures = []
        for process, created in identities:
            try:
                if process.create_time() == created:
                    process.terminate()
            except psutil.NoSuchProcess:
                pass
            except psutil.Error as exc:
                failures.append(dict(pid=process.pid, error=str(exc)))
        _, alive = psutil.wait_procs([p for p, _ in identities], timeout=10)
        receipt.update(status='stopped' if not alive and not failures else 'stop_incomplete',
                       alive_pids=[p.pid for p in alive], errors=failures)
        save(folder / 'driver_guard_stop.json', receipt)
        if receipt['status'] == 'stopped':
            prior = receipt.get('queue_before_stop', {})
            save(OUT / 'queue_status.json', dict(status='stopped_for_driver_error_burst',
                 completed=prior.get('completed'), total=12, active_models=[],
                 previous_active_models=prior.get('active_models', []), max_workers=2,
                 automatic_restart_allowed=False, guard_record=str(folder / 'driver_guard_stop.json')))
        return receipt
    finally:
        # Do not leave a surviving process suspended if evidence capture failed.
        for process in suspended:
            try:
                if process.is_running():
                    process.resume()
            except psutil.Error:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue-pid', type=int)
    parser.add_argument('--recovery-dir', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(events(), ensure_ascii=True), flush=True)
        return
    if args.queue_pid is None or args.recovery_dir is None:
        parser.error('--queue-pid and --recovery-dir are required')
    folder = args.recovery_dir.resolve()
    if not folder.is_relative_to((OUT / 'recoveries').resolve()):
        raise ValueError('Recovery directory must remain under this experiment')
    queue = psutil.Process(args.queue_pid)
    if not any(Path(x).name == 'run_stgcnpp_ab.py' for x in queue.cmdline()):
        raise ValueError('Target is not the A/B queue')
    created = queue.create_time()
    consecutive = 0
    while queue.is_running() and queue.create_time() == created:
        start = time.monotonic()
        try:
            observation = events()
            consecutive = consecutive + 1 if observation['count'] >= 100 else 0
            observation.update(status='observing', queue_pid=queue.pid,
                               consecutive_high_windows=consecutive)
            save(folder / 'driver_guard_status.json', observation)
            with (folder / 'driver_guard_events.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(observation, ensure_ascii=True) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            if observation['count'] >= 500 or consecutive >= 2:
                stopped = stop_queue(queue, folder, observation)
                save(folder / 'driver_guard_status.json', stopped)
                print(json.dumps(stopped, ensure_ascii=True), flush=True)
                return
        except Exception as exc:
            consecutive = 0
            save(folder / 'driver_guard_status.json', dict(status='monitor_error',
                 at=dt.datetime.now().isoformat(), error=str(exc), queue_pid=queue.pid))
        time.sleep(max(1, 60 - (time.monotonic() - start)))
    save(folder / 'driver_guard_status.json', dict(status='queue_exited', at=dt.datetime.now().isoformat()))


if __name__ == '__main__':
    main()
