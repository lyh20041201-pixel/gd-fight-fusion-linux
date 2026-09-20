"""模拟摄像头。

生成带教室背景与移动"人员"的合成画面，并在 meta 中给出真值框，
供模拟检测器使用（真实模式下由 YOLO 产生框，接口一致）。
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

import cv2
import numpy as np

from .base import CameraSource, Frame


class _Walker:
    """一个模拟人员：在画面内缓慢走动，偶尔离开。"""

    def __init__(self, rng: random.Random, width: int, height: int) -> None:
        self.rng = rng
        self.width = width
        self.height = height
        self.x = rng.uniform(0.1, 0.9) * width
        self.y = rng.uniform(0.45, 0.85) * height
        self.vx = rng.uniform(-18, 18)
        self.vy = rng.uniform(-6, 6)
        self.scale = rng.uniform(0.75, 1.15)
        self.phase = rng.uniform(0, 6.28)

    def step(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.x < 40 or self.x > self.width - 40:
            self.vx *= -1
            self.x = min(max(self.x, 40), self.width - 40)
        if self.y < self.height * 0.4 or self.y > self.height - 30:
            self.vy *= -1
            self.y = min(max(self.y, self.height * 0.4), self.height - 30)
        if self.rng.random() < 0.01:
            self.vx = self.rng.uniform(-22, 22)
            self.vy = self.rng.uniform(-8, 8)

    def bbox(self, t: float) -> list[float]:
        h = 150 * self.scale * (0.85 + 0.15 * (self.y / max(1.0, self.height)))
        w = h * 0.38
        bob = 3.0 * math.sin(t * 4 + self.phase)
        x1 = self.x - w / 2
        y1 = self.y - h + bob
        return [x1, y1, x1 + w, y1 + h]


class SimulatedCamera(CameraSource):
    """合成画面摄像头。"""

    def __init__(
        self,
        camera_id: str,
        name: str,
        index: int,
        width: int = 640,
        height: int = 480,
        fps: int = 12,
        seed: int | None = None,
        base_people: int = 3,
    ) -> None:
        super().__init__(camera_id, name, f"sim:{index}")
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._rng = random.Random(seed if seed is not None else 1000 + index)
        self._frame_index = 0
        self._last_read = 0.0
        self._offline_until: float | None = None
        self._obstructed_until: float | None = None
        self._walkers = [
            _Walker(self._rng, width, height) for _ in range(base_people)
        ]
        self._background = self._make_background()
        self._base_people = base_people
        self._target_people = base_people

    # ---------- 生命周期 ----------
    def open(self) -> bool:
        self._opened = True
        self.last_error = None
        return True

    def close(self) -> None:
        self._opened = False

    @property
    def simulated(self) -> bool:
        return True

    # ---------- 场景控制 ----------
    def set_offline(self, seconds: float) -> None:
        self._offline_until = time.time() + seconds

    def set_obstructed(self, seconds: float) -> None:
        self._obstructed_until = time.time() + seconds

    def set_people(self, count: int) -> None:
        count = max(0, min(30, int(count)))
        self._target_people = count
        while len(self._walkers) < count:
            self._walkers.append(_Walker(self._rng, self.width, self.height))
        while len(self._walkers) > count:
            self._walkers.pop()

    def recover(self) -> None:
        self._offline_until = None
        self._obstructed_until = None

    @property
    def people(self) -> int:
        return self._target_people

    @property
    def base_people(self) -> int:
        return self._base_people

    # ---------- 读帧 ----------
    def read(self) -> Frame | None:
        now = time.time()
        if self._offline_until and now < self._offline_until:
            self.last_error = "模拟摄像头掉线"
            return None
        if self._offline_until and now >= self._offline_until:
            self._offline_until = None
            self.last_error = None

        interval = 1.0 / max(1, self.fps)
        wait = self._last_read + interval - now
        if wait > 0:
            time.sleep(min(wait, interval))
            now = time.time()
        dt = min(0.5, now - self._last_read) if self._last_read else interval
        self._last_read = now
        self._frame_index += 1

        # 人数缓慢波动（约每分钟一次，且围绕基线小幅变化，避免 Track ID 频繁翻新）
        if self._rng.random() < 0.0015:
            delta = self._rng.choice([-1, 1])
            candidate = self._target_people + delta
            low, high = max(0, self._base_people - 2), self._base_people + 3
            self.set_people(min(high, max(low, candidate)))

        image = self._background.copy()
        truth: list[dict[str, Any]] = []
        for walker in self._walkers:
            walker.step(dt)
            box = walker.bbox(now)
            self._draw_person(image, box)
            truth.append({"bbox": box, "confidence": round(self._rng.uniform(0.62, 0.94), 2)})

        image = self._add_noise(image)
        obstructed = bool(self._obstructed_until and now < self._obstructed_until)
        if obstructed:
            image = (image * 0.06).astype(np.uint8)
            truth = []
        elif self._obstructed_until:
            self._obstructed_until = None

        self._stamp(image, now)
        return Frame(
            image=image,
            timestamp=now,
            index=self._frame_index,
            meta={"truth": truth, "simulated": True, "obstructed": obstructed},
        )

    # ---------- 绘制 ----------
    def _make_background(self) -> np.ndarray:
        img = np.full((self.height, self.width, 3), 38, dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (self.width, int(self.height * 0.38)), (58, 55, 52), -1)
        cv2.rectangle(
            img,
            (int(self.width * 0.08), int(self.height * 0.06)),
            (int(self.width * 0.55), int(self.height * 0.3)),
            (72, 84, 74),
            -1,
        )
        cv2.putText(
            img,
            "BLACKBOARD",
            (int(self.width * 0.12), int(self.height * 0.2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (120, 140, 120),
            1,
            cv2.LINE_AA,
        )
        rows, cols = 4, 5
        for r in range(rows):
            for c in range(cols):
                x = int(self.width * (0.06 + c * 0.19))
                y = int(self.height * (0.5 + r * 0.12))
                cv2.rectangle(img, (x, y), (x + 70, y + 26), (70, 66, 60), -1)
                cv2.rectangle(img, (x, y), (x + 70, y + 26), (48, 46, 42), 1)
        return img

    def _draw_person(self, img: np.ndarray, box: list[float]) -> None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cx = (x1 + x2) // 2
        head_r = max(6, (x2 - x1) // 3)
        body_top = y1 + head_r * 2
        cv2.circle(img, (cx, y1 + head_r), head_r, (168, 158, 148), -1)
        cv2.rectangle(img, (x1 + 2, body_top), (x2 - 2, y2), (96, 112, 148), -1)
        cv2.line(img, (cx, y2 - 4), (cx - 8, y2 + 6), (70, 82, 108), 3)
        cv2.line(img, (cx, y2 - 4), (cx + 8, y2 + 6), (70, 82, 108), 3)

    def _add_noise(self, img: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, 3.5, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def _stamp(self, img: np.ndarray, now: float) -> None:
        text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        cv2.putText(
            img,
            f"SIM {self.camera_id} {text}",
            (8, self.height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
