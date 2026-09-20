import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

from scripts.guard_fight_fusion import OwnedJob, PressureMonitor, memory_snapshot


def snapshot(available=16., commit=20., gpu_free=4000.):
    return dict(available_gib=available, commit_free_gib=commit, gpu=dict(free_mib=gpu_free))


def test_memory_reserve_blocks_start_and_stops_before_exhaustion():
    monitor = PressureMonitor()
    assert monitor.check(snapshot(available=7), starting=True)
    assert monitor.check(snapshot(commit=7), starting=True)
    assert monitor.check(snapshot(), starting=True) is None
    assert monitor.check(snapshot(available=3.9))
    assert monitor.check(snapshot(commit=3.9))


def test_transient_pressure_recovers_but_sustained_pressure_stops():
    monitor = PressureMonitor()
    assert monitor.check(snapshot(available=5)) is None
    assert monitor.check(snapshot(available=5)) is None
    assert monitor.check(snapshot()) is None
    assert monitor.check(snapshot(commit=5)) is None
    assert monitor.check(snapshot(commit=5)) is None
    assert monitor.check(snapshot(commit=5))


def test_gpu_spike_tolerated_but_sustained_exhaustion_stops():
    monitor = PressureMonitor()
    assert monitor.check(snapshot(gpu_free=100)) is None
    assert monitor.check(snapshot()) is None
    assert monitor.check(snapshot(gpu_free=100)) is None
    assert monitor.check(snapshot(gpu_free=100)) is None
    assert monitor.check(snapshot(gpu_free=100))


@pytest.mark.skipif(os.name != 'nt', reason='Windows job integration')
def test_real_windows_memory_snapshot_has_commit_and_pool_counters():
    state = memory_snapshot()
    assert state['available_gib'] > 0
    assert state['commit_free_gib'] > 0
    assert state['kernel_paged_gib'] > 0
    assert state['kernel_nonpaged_gib'] > 0


@pytest.mark.skipif(os.name != 'nt', reason='Windows job integration')
def test_pressure_kills_owned_descendants_preserving_checkpoint_and_unrelated_process(tmp_path):
    checkpoint = tmp_path / 'resume.pt'
    checkpoint.write_bytes(b'previous atomic checkpoint')
    child_pid_path = tmp_path / 'child.pid'
    helper = tmp_path / 'helper.py'
    helper.write_text(
        'import subprocess, sys, time\n'
        'from pathlib import Path\n'
        'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])\n'
        f'Path({str(child_pid_path)!r}).write_text(str(p.pid))\n'
        'time.sleep(60)\n', encoding='utf-8')
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                 creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        with OwnedJob() as job, (tmp_path / 'log.txt').open('wb') as log:
            parent = job.launch([helper], log=log, cwd=tmp_path)
            deadline = time.monotonic() + 10
            while not child_pid_path.exists() and time.monotonic() < deadline:
                time.sleep(.05)
            assert child_pid_path.exists()
            child = psutil.Process(int(child_pid_path.read_text()))
            assert child.is_running()
            assert PressureMonitor().check(snapshot(available=3))
            job.close()
            parent.wait(timeout=10)
            child.wait(timeout=10)
            assert unrelated.poll() is None
            assert checkpoint.read_bytes() == b'previous atomic checkpoint'
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=10)
