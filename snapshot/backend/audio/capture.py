"""Bounded PCM capture. Audio uses the same wall-clock timeline as camera frames."""
from __future__ import annotations

import io
import math
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AudioEvidence:
    wav: bytes | None
    metadata: dict


class AudioRingBuffer:
    def __init__(self, seconds: float = 15, sample_rate: int = 16000):
        self.seconds, self.sample_rate = seconds, sample_rate
        self._chunks = deque()
        self._lock = threading.Lock()

    def clear(self):
        with self._lock:
            self._chunks.clear()

    def offer(self, pcm: bytes, start: float, valid: bool = True):
        if not math.isfinite(start) or not pcm or len(pcm) % 2:
            return
        # Bound both duration and object count, including broken device timestamps.
        pcm = pcm[: int(self.seconds * self.sample_rate) * 2]
        with self._lock:
            if self._chunks and start < self._chunks[-1][0]:
                self._chunks.clear()
            self._chunks.append((start, bytes(pcm), valid))
            cutoff = start + len(pcm) / (2 * self.sample_rate) - self.seconds
            while self._chunks and (self._chunks[0][0] + len(self._chunks[0][1]) / (2 * self.sample_rate) <= cutoff
                                    or len(self._chunks) > int(self.seconds * 100) + 2):
                self._chunks.popleft()

    def extract(self, start: float, end: float) -> AudioEvidence:
        if not all(math.isfinite(x) for x in (start, end)) or not 0 < end - start <= self.seconds:
            raise ValueError("音频证据时间窗无效或超过缓存长度")
        count = round((end - start) * self.sample_rate)
        samples = np.zeros(count, dtype='<i2')
        present = np.zeros(count, dtype=bool)
        with self._lock:
            chunks = list(self._chunks)
        for ts, pcm, valid in chunks:
            if not valid:
                continue
            data = np.frombuffer(pcm, dtype='<i2')
            offset = round((ts - start) * self.sample_rate)
            lo, hi = max(0, offset), min(count, offset + len(data))
            if hi > lo:
                samples[lo:hi] = data[lo - offset:hi - offset]
                present[lo:hi] = True
        coverage = float(present.mean()) if count else 0
        meta = {"status": "ready" if coverage >= .98 else "partial" if coverage else "unavailable",
                "start": start, "end": end, "duration_seconds": end-start,
                "sample_rate": self.sample_rate, "channels": 1, "format": "wav",
                "coverage": round(coverage, 4), "path": None,
                "reason": None if coverage >= .98 else "录音不完整或时间窗内没有采集数据"}
        if not coverage:
            return AudioEvidence(None, meta)
        changes = np.diff(np.r_[False, ~present, False].astype(np.int8))
        meta["missing_intervals"] = [[round(a/self.sample_rate, 4), round(b/self.sample_rate, 4)]
                                     for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]
        meta["clipping_ratio"] = round(float((np.abs(samples.astype(np.int32)[present]) >= 32760).mean()), 4)
        output = io.BytesIO()
        with wave.open(output, 'wb') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(self.sample_rate)
            f.writeframes(samples.tobytes())
        return AudioEvidence(output.getvalue(), meta)


class MicrophoneCapture:
    def __init__(self, settings, runtime):
        self.settings, self.runtime = settings, runtime
        self.ring = AudioRingBuffer(settings.audio_buffer_seconds, settings.audio_sample_rate)
        self._stop = threading.Event()
        self._thread = None
        self.online = False
        self.last_error = None
        self.last_sample_at = None
        self.device_name = ""
        self.level_dbfs = None
        self.startup_simulation = False

    @property
    def simulated(self):
        # A saved mode change cannot turn a running simulation into physical capture.
        return self.startup_simulation or self.runtime.simulation_mode

    @staticmethod
    def devices():
        try:
            import sounddevice as sd
            hosts = sd.query_hostapis()
            return {"devices": [{"index": i, "name": d['name'], "hostapi": hosts[d['hostapi']]['name'],
                                 "channels": d['max_input_channels']}
                                for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0],
                    "error": None}
        except Exception as exc:
            return {"devices": [], "error": f"麦克风枚举失败：{type(exc).__name__}"}

    def start(self):
        if self.simulated or not self.runtime.get('audio_enabled', False):
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name='microphone', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=4)
        self.online = False
        self.last_sample_at = None
        self.ring.clear()

    def _run(self):
        while not self._stop.is_set():
            try:
                import sounddevice as sd
                index = self.runtime.get('audio_device_index', -1)
                device = None if index == -1 else index
                self.device_name = str(sd.query_devices(device, 'input')['name'])
                self.ring.clear()
                self.last_sample_at = None
                wall_anchor, mono_anchor = time.time(), time.monotonic()

                def callback(indata, frames, timing, status):
                    if self._stop.is_set():
                        return
                    # Correct ADC/driver latency; monotonic anchor prevents per-block wall-clock jumps.
                    ts = wall_anchor + time.monotonic() - mono_anchor + timing.inputBufferAdcTime - timing.currentTime
                    self.ring.offer(bytes(indata), ts, valid=not bool(status))
                    self.last_sample_at = ts + frames / self.settings.audio_sample_rate
                    self.online = True
                    self.last_error = "音频采集丢帧" if status else None
                    values = np.frombuffer(indata, dtype='<i2').astype(np.float32) / 32768
                    self.level_dbfs = round(20 * math.log10(max(1e-6, float(np.sqrt(np.mean(values**2))))), 1)

                with sd.RawInputStream(device=device, samplerate=self.settings.audio_sample_rate,
                                       channels=1, dtype='int16', blocksize=self.settings.audio_sample_rate // 10,
                                       callback=callback) as stream:
                    while not self._stop.wait(.25):
                        if not stream.active or time.time()-(self.last_sample_at or wall_anchor) > 2:
                            raise RuntimeError('microphone stopped')
            except Exception as exc:
                self.online = False
                self.last_error = f"麦克风不可用，正在重连：{type(exc).__name__}"
                self._stop.wait(3)
        self.online = False

    def status(self):
        enabled = bool(self.runtime.get('audio_enabled', False))
        return {"enabled": enabled, "online": self.online and bool(self.last_sample_at and time.time()-self.last_sample_at < 2),
                "device_index": self.runtime.get('audio_device_index', -1), "device_name": self.device_name,
                "sample_rate": self.settings.audio_sample_rate, "buffer_seconds": self.settings.audio_buffer_seconds,
                "camera_ids": self.settings.audio_camera_ids, "level_dbfs": self.level_dbfs,
                "last_sample_at": self.last_sample_at,
                "error": "模拟模式不采集真实声音" if self.simulated else self.last_error}

    def evidence(self, start, end, camera_id=None):
        reason = None
        if self.simulated:
            reason = "模拟模式不采集真实声音"
        elif not self.runtime.get('audio_enabled', False):
            reason = "麦克风已停用"
        elif self.settings.audio_camera_ids and camera_id not in self.settings.audio_camera_ids:
            reason = "该摄像头未绑定此麦克风"
        if reason:
            return AudioEvidence(None, {"status": "unavailable", "start": start, "end": end,
                                        "coverage": 0, "path": None, "reason": reason})
        evidence = self.ring.extract(start, end)
        evidence.metadata.update(device_name=self.device_name, source="room_microphone",
                                 camera_id=camera_id, source_attribution="同室环境声，不能直接归属到某个人")
        return evidence


def save_audio(evidence: AudioEvidence, folder: Path, root: Path):
    metadata = dict(evidence.metadata)
    if evidence.wav:
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / 'audio.wav'
        target.write_bytes(evidence.wav)
        metadata['path'] = target.relative_to(root).as_posix()
    return metadata
