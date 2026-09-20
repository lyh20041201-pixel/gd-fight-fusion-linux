"""Video predictions and shared sampling; live evidence gates live separately."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import cv2


@dataclass
class VideoPrediction:
    event_type: str
    score: float
    start: float
    end: float
    model_version: str
    guard_version: str | None = None
    diagnostics: dict | None = None


def sample_window(frames, count=16, seconds=4.0):
    """Uniformly sample timestamps, retaining actual times for padded frames."""
    if not frames:
        raise ValueError("empty video window")
    frames = sorted(frames, key=lambda x: x[0])
    end = frames[-1][0]
    frames = [f for f in frames if f[0] >= end - seconds]
    times = np.asarray([f[0] for f in frames])
    indices = [int(np.argmin(abs(times - t))) for t in np.linspace(max(times[0], end-seconds), end, count)]
    return [frames[i] for i in indices]


def clip_tensor(images):
    import torch
    # Letterbox preserves full scene; identical preprocessing in train and runtime.
    out = []
    for image in images:
        h, w = image.shape[:2]
        scale = 112 / max(h, w)
        resized = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))))
        canvas = np.zeros((112, 112, 3), np.uint8)
        y, x = (112-resized.shape[0])//2, (112-resized.shape[1])//2
        canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
        out.append(canvas[..., ::-1].copy())
    tensor = torch.from_numpy(np.stack(out)).float().permute(3, 0, 1, 2)/255
    mean = torch.tensor([.43216,.394666,.37645])[:,None,None,None]
    std = torch.tensor([.22803,.22145,.216989])[:,None,None,None]
    return (tensor-mean)/std


class VideoEventDetector:
    def __init__(self, checkpoint: str, device="cuda:0"):
        import torch
        from torchvision.models.video import r3d_18
        self.device = device
        self.path = Path(checkpoint)
        saved = torch.load(self.path, map_location="cpu", weights_only=True)
        self.labels = saved["labels"]
        self.threshold = saved["threshold"]
        self.model = r3d_18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.labels))
        self.model.load_state_dict(saved["state_dict"])
        self.model.to(device).eval()
        self.version = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def predict(self, frames):
        import torch
        selected = sample_window(frames)
        with torch.inference_mode():
            probs = self.model(clip_tensor([x[1] for x in selected]).unsqueeze(0).to(self.device)).softmax(1)[0].cpu().tolist()
        fall = sum(p for label,p in zip(self.labels,probs) if label in ("fall", "fallen"))
        fight = sum(p for label,p in zip(self.labels,probs) if label == "fight")
        return [VideoPrediction(kind, score, selected[0][0], selected[-1][0], self.version)
                for kind, score in (("person_fall",fall),("suspected_fight",fight)) if score >= self.threshold]
