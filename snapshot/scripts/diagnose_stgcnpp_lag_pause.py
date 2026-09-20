"""Temporarily suspend only this experiment for a user-authorized 120s comparison."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/video_events/skeleton_stgcnpp_ab'


def save(path, record):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(record, handle, ensure_ascii=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sample():
    memory = psutil.virtual_memory()
    result = dict(at=datetime.datetime.now().isoformat(),
                  ram_percent=memory.percent, available_gib=memory.available / 2**30)
    try:
        result['gpu_csv'] = subprocess.check_output([
            'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'], text=True, timeout=6).strip()
        result['gpu_columns'] = ['utilization_percent', 'memory_used_mib', 'temperature_c', 'power_w']
    except Exception as error:
        result['gpu_error'] = str(error)
    return result


def main():
    queue = json.loads((OUT / 'queue_status.json').read_text(encoding='utf-8'))
    assert queue['status'] == 'training' and len(queue['active_models']) == 2
    workers = [psutil.Process(item['pid']) for item in queue['active_models']]
    parent_ids = {worker.ppid() for worker in workers}
    assert len(parent_ids) == 1
    parent = psutil.Process(parent_ids.pop())
    assert 'scripts/run_stgcnpp_ab.py' in parent.cmdline()
    for worker in workers:
        assert 'scripts/train_stgcnpp_ab.py' in worker.cmdline()
    destination = ROOT / 'results/system_diagnostics' / datetime.datetime.now().strftime('lag_pause_%Y%m%d_%H%M%S')
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / 'observation.json'
    record = dict(status='preparing', requested_seconds=120,
                  processes=[dict(pid=p.pid, created_at=p.create_time(), command=p.cmdline())
                             for p in [parent, *workers]],
                  active_models=queue['active_models'], before=sample(), during=[],
                  training_files_modified=False,
                  timing_note='This pause adds wall time to current epoch histories; do not use those epochs as clean throughput measurements.')
    save(path, record)
    suspended = []
    try:
        for process in [parent, *workers]:
            process.suspend()
            suspended.append(process)
        started = time.monotonic()
        record.update(status='paused_for_120_second_diagnostic', paused_at=datetime.datetime.now().isoformat())
        save(path, record)
        print(json.dumps(dict(event='PAUSED', path=str(path), paused_at=record['paused_at'],
                              pids=[p.pid for p in suspended]), ensure_ascii=True), flush=True)
        for offset in [10, 60, 110]:
            time.sleep(max(0, started + offset - time.monotonic()))
            observation = sample()
            observation['offset_seconds'] = round(time.monotonic() - started, 3)
            record['during'].append(observation)
            save(path, record)
            print(json.dumps(dict(event='PAUSE_SAMPLE', **observation), ensure_ascii=True), flush=True)
        time.sleep(max(0, started + 120 - time.monotonic()))
    finally:
        failures = []
        for process in reversed(suspended):
            try:
                process.resume()
            except psutil.Error as error:
                failures.append(dict(pid=process.pid, error=str(error)))
        record.update(status='resumed' if not failures else 'resume_needs_attention',
                      resumed_at=datetime.datetime.now().isoformat(), resume_failures=failures)
        if 'started' in locals():
            record['actual_pause_seconds'] = time.monotonic() - started
        save(path, record)
        print(json.dumps(dict(event='RESUMED', path=str(path), status=record['status'],
                              resumed_at=record['resumed_at'], failures=failures), ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
