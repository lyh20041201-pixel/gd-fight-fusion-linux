"""Compare a named USB camera's capture backend/format without changing GD settings.

Run only while that device is released by the website and other camera apps.
Each mode runs in a child process so a stuck driver has a bounded timeout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def capture(args):
    import cv2

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    backend, fourcc = args.mode.split(":")
    cap = cv2.VideoCapture(args.index, getattr(cv2, "CAP_" + backend))
    result = dict(index=args.index, requested_backend=backend, requested_fourcc=fourcc)
    try:
        if not cap.isOpened():
            raise RuntimeError("Camera could not be opened; check device selection/other camera apps")
        applied = {}
        if fourcc != "AUTO":
            applied["fourcc"] = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        for name, prop, value in (("width", cv2.CAP_PROP_FRAME_WIDTH, 640),
                                  ("height", cv2.CAP_PROP_FRAME_HEIGHT, 480),
                                  ("fps", cv2.CAP_PROP_FPS, 15)):
            applied[name] = cap.set(prop, value)
        frames = []
        started = time.monotonic()
        for i in range(30):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Camera returned no frame")
            if i >= 20:
                frames.append(frame)
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        name = args.mode.replace(":", "_")
        destination = root / (name + ".png")
        cv2.imwrite(str(destination), frames[-1])
        result.update(ok=True, backend=cap.getBackendName(), applied=applied,
                      width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      fps=cap.get(cv2.CAP_PROP_FPS),
                      fourcc="".join(chr((code >> (8 * i)) & 255) for i in range(4)),
                      observed_fps=round(30 / (time.monotonic() - started), 2),
                      image=str(destination))
    except Exception as exc:
        result.update(ok=False, error=str(exc))
    finally:
        cap.release()
    (root / (args.mode.replace(":", "_") + ".json")).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["DSHOW:AUTO", "DSHOW:MJPG", "MSMF:AUTO"])
    args = parser.parse_args()
    if args.mode:
        capture(args)
        return
    for mode in ("DSHOW:AUTO", "DSHOW:MJPG", "MSMF:AUTO"):
        try:
            subprocess.run([sys.executable, __file__, "--index", str(args.index),
                            "--output", args.output, "--mode", mode], timeout=25, check=True)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            print(json.dumps(dict(mode=mode, ok=False, error=str(exc))), flush=True)


if __name__ == "__main__":
    main()
