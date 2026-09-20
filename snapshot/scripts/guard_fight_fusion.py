"""Run the unchanged, sealed fight queue with an independent Windows watchdog.

The watchdog never loads CUDA or modifies experiment signatures. A Windows job
owns only the queue it launches and its descendants; closing the watchdog also
closes that job. Emergency stops keep the trainer's last atomic checkpoint,
and may lose work since that checkpoint. They never restart automatically.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/fight_fusion_v1"
GIB = 1024**3
POLICY = dict(interval_seconds=5, start_available_gib=8, start_commit_free_gib=8,
              hard_available_gib=4, hard_commit_free_gib=4,
              soft_available_gib=6, soft_commit_free_gib=6,
              soft_samples=3, gpu_free_mib=256, gpu_low_samples=3)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".guard.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(40):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


class PerformanceInfo(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal",
            "PhysicalAvailable", "SystemCache", "KernelTotal", "KernelPaged",
            "KernelNonpaged", "PageSize")
    ] + [(name, wintypes.DWORD) for name in ("HandleCount", "ProcessCount", "ThreadCount")]


def memory_snapshot():
    info = PerformanceInfo()
    info.cb = ctypes.sizeof(info)
    api = ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo
    api.argtypes = [ctypes.POINTER(PerformanceInfo), wintypes.DWORD]
    api.restype = wintypes.BOOL
    if not api(ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    size = info.PageSize / GIB
    return dict(available_gib=info.PhysicalAvailable * size,
                commit_free_gib=(info.CommitLimit - info.CommitTotal) * size,
                committed_gib=info.CommitTotal * size,
                kernel_paged_gib=info.KernelPaged * size,
                kernel_nonpaged_gib=info.KernelNonpaged * size)


def gpu_snapshot():
    result = subprocess.run([
        "nvidia-smi", "--id=0",
        "--query-gpu=memory.free,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits"], capture_output=True, text=True,
        timeout=4, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    values = [float(v.strip()) for v in result.stdout.strip().split(",")]
    return dict(zip(("free_mib", "used_mib", "total_mib", "utilization", "temperature"), values))


class PressureMonitor:
    def __init__(self):
        self.host_low = self.gpu_low = 0

    def check(self, snapshot, *, starting=False):
        available, commit = snapshot["available_gib"], snapshot["commit_free_gib"]
        if starting and (available < POLICY["start_available_gib"] or
                         commit < POLICY["start_commit_free_gib"]):
            return "Startup requires at least 8 GiB of physical and commit headroom"
        if available < POLICY["hard_available_gib"] or commit < POLICY["hard_commit_free_gib"]:
            return "Physical or commit headroom below 4 GiB; emergency stop"
        self.host_low = self.host_low + 1 if (
            available < POLICY["soft_available_gib"] or commit < POLICY["soft_commit_free_gib"]
        ) else 0
        self.gpu_low = self.gpu_low + 1 if snapshot["gpu"]["free_mib"] < POLICY["gpu_free_mib"] else 0
        if starting and self.gpu_low:
            return "Insufficient free dedicated GPU memory at startup"
        if self.host_low >= POLICY["soft_samples"]:
            return "Physical or commit headroom below 6 GiB for three consecutive samples"
        if self.gpu_low >= POLICY["gpu_low_samples"]:
            return "Dedicated GPU memory headroom below 256 MiB for three consecutive samples"
        return None


class BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IOCounters)] + [
        (name, ctypes.c_size_t) for name in (
            "ProcessMemoryLimit", "JobMemoryLimit", "PeakProcessMemoryUsed", "PeakJobMemoryUsed")]


class OwnedJob:
    """Only launch Python children after assignment to a kill-on-close job."""
    def __init__(self):
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            "OpenProcess": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = args, result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def launch(self, arguments, *, log, cwd=ROOT):
        # The child blocks before importing training code or creating descendants.
        bootstrap = ("import runpy,sys; gate=sys.stdin.buffer.read(1); sys.stdin.close(); "
                     "assert gate==b'1', 'Missing job assignment'; "
                     "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')")
        process = subprocess.Popen([sys.executable, "-u", "-c", bootstrap, *map(str, arguments)],
                                   cwd=cwd, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        handle = self.api.OpenProcess(0x0101, False, process.pid)  # SET_QUOTA | TERMINATE
        try:
            if not handle or not self.api.AssignProcessToJobObject(self.handle, handle):
                raise ctypes.WinError(ctypes.get_last_error())
            process.stdin.write(b"1")
            process.stdin.flush()
            process.stdin.close()
        except BaseException:
            process.kill()
            process.wait(timeout=10)
            raise
        finally:
            if handle:
                self.api.CloseHandle(handle)
        return process

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@contextmanager
def guard_lock():
    import msvcrt
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "resource_guard.lock").open("a+b") as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def check_no_existing_training():
    names = ("run_fight_fusion.py", "train_fight_fusion.py", "fit_fight_fusion.py")
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if not (process.info["name"] or "").lower().startswith("python"):
            continue
        args = process.info["cmdline"] or []
        if any(Path(arg).name in names for arg in args):
            raise RuntimeError(f"Existing fight training process {process.pid}; refusing a duplicate queue")


def process_snapshot(pid):
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []
    rows = []
    for process in processes:
        try:
            memory = process.memory_info()
            rows.append(dict(pid=process.pid, rss_gib=memory.rss / GIB,
                             private_gib=memory.private / GIB))
        except psutil.NoSuchProcess:
            pass
    return rows


def run():
    if os.name != "nt":
        raise RuntimeError("This watchdog requires Windows job objects")
    with guard_lock():
        check_no_existing_training()
        monitor = PressureMonitor()
        initial = dict(memory_snapshot(), gpu=gpu_snapshot())
        reason = monitor.check(initial, starting=True)
        if reason:
            raise RuntimeError(reason)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        folder = OUT / "resource_guard" / stamp
        folder.mkdir(parents=True)
        record = dict(pid=os.getpid(), started=time.time(), policy=POLICY,
                      supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      folder=str(folder), automatic_restart=False)
        write_json(folder / "launch.json", dict(record, initial=initial))
        with OwnedJob() as job, (folder / "queue.log").open("ab") as log:
            process = job.launch([ROOT / "scripts/run_fight_fusion.py"], log=log)
            record["queue_pid"] = process.pid
            snapshot = initial
            try:
                with (folder / "samples.jsonl").open("a", encoding="utf-8") as samples:
                    while process.poll() is None:
                        snapshot = dict(memory_snapshot(), gpu=gpu_snapshot(),
                                        processes=process_snapshot(process.pid))
                        reason = monitor.check(snapshot)
                        status = dict(record, status="running", updated=time.time(), resources=snapshot)
                        if reason:
                            raise RuntimeError(reason)
                        samples.write(json.dumps(status) + "\n")
                        samples.flush()
                        write_json(OUT / "resource_guard_status.json", status)
                        try:
                            process.wait(timeout=POLICY["interval_seconds"])
                        except subprocess.TimeoutExpired:
                            pass
                if process.returncode:
                    raise RuntimeError(f"Queue exited with code {process.returncode}; see {folder / 'queue.log'}")
            except BaseException as exc:
                # Stop descendants before writing status: the queue cannot launch a new stage.
                job.close()
                process.wait(timeout=10)
                state = dict(record, status="attention_required", updated=time.time(),
                             reason=str(exc), resources=snapshot,
                             last_complete_checkpoint_preserved=True)
                write_json(OUT / "resource_guard_status.json", state)
                write_json(folder / "stop.json", state)
                write_json(OUT / "queue_status.json", dict(state, stage="resource_guard"))
                raise
            write_json(OUT / "resource_guard_status.json",
                       dict(record, status="complete", updated=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read resources without launching training")
    args = parser.parse_args()
    if args.check:
        snapshot = dict(memory_snapshot(), gpu=gpu_snapshot())
        print(json.dumps(dict(resources=snapshot, policy=POLICY,
                              startup_block=PressureMonitor().check(snapshot, starting=True)), indent=2))
    else:
        run()
