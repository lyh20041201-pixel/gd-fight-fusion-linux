"""应用级配置。

所有配置来自环境变量 / .env 文件；密钥（DASHSCOPE_API_KEY）只从环境变量读取，
不写入数据库、日志与前端。可在运行时由用户修改的业务参数（阈值、ROI 等）
保存在 SQLite 的 settings 表中，见 backend/config/runtime.py。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- 运行模式 ----------
    # 默认真实模式：没有接入硬件时界面显示离线/无数据，不生成任何模拟数据
    simulation_mode: bool = False

    # ---------- 服务 ----------
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: str = "INFO"
    # 逗号分隔的额外来源白名单（CORS 与 WebSocket 共用）。绑定到非本机地址时
    # 必须在此显式声明前端来源，并同时为接口补充鉴权。
    extra_allowed_origins: str = ""

    # ---------- 串口 ----------
    serial_port: str = ""
    serial_baudrate: int = 115200
    serial_auto_discover: bool = True
    serial_reconnect_seconds: float = 3.0
    command_ack_timeout: float = 2.0
    command_max_retries: int = 3
    heartbeat_timeout_seconds: float = 15.0

    # ---------- 摄像头 ----------
    camera_device_indices: str = "0,1,2,3"
    simulated_camera_count: int = 4
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 15
    camera_backend: Literal['AUTO', 'DSHOW', 'MSMF'] = 'AUTO'
    camera_fourcc: Literal['AUTO', 'MJPG', 'YUY2'] = 'AUTO'
    stream_fps: int = 8
    frame_buffer_seconds: float = 5.0

    # ---------- 视觉 ----------
    vision_enabled: bool = True
    yolo_model_path: str = "models/yolov8n.pt"
    yolo_confidence: float = 0.1
    yolo_device: str = "cpu"
    # 默认开启人脸模糊：教室场景大概率涉及未成年人影像，未显式关闭前不留清晰人脸。
    face_blur_enabled: bool = True

    # ---------- Qwen ----------
    dashscope_api_key: str = Field(default="", repr=False)
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: Literal['qwen3.5-omni-flash', 'qwen3.5-omni-plus'] = "qwen3.5-omni-flash"
    # 默认关闭：启用后会把教室画面提交给外部模型，需由部署方显式开启。
    qwen_enabled: bool = False
    qwen_timeout_seconds: float = Field(default=45, ge=1, le=180)
    qwen_max_retries: int = Field(default=1, ge=0, le=3)
    qwen_cooldown_seconds: float = Field(default=30, ge=0, le=3600)
    qwen_max_queue: int = Field(default=16, ge=1, le=64)
    qwen_max_frames: int = Field(default=8, ge=1, le=8)
    qwen_max_output_tokens: int = Field(default=2048, ge=256, le=4096)

    # One room microphone; optional camera IDs restrict which views share its audio.
    audio_enabled: bool = True
    audio_device_index: int = Field(default=-1, ge=-1, le=1024)
    audio_sample_rate: Literal[16000, 24000, 48000] = 16000
    audio_buffer_seconds: float = Field(default=15, ge=10, le=30)
    audio_pre_seconds: float = Field(default=5, ge=3, le=10)
    audio_post_seconds: float = Field(default=2.5, ge=0, le=5)
    audio_camera_ids: list[str] = Field(default_factory=list)
    # Temporal models are enabled only when explicitly configured with trained checkpoints.
    # Rebuilt checkpoints stay disabled until training/evaluation is complete.
    video_event_checkpoints: list[str] = Field(default_factory=list)
    video_event_device: str = "cuda:0"
    # Live actions use a separate worker so deployment keeps the training runtime.
    live_actions_enabled: bool = False
    live_actions_manifest: str = "config/live_actions.json"
    live_actions_python: str = ""

    # ---------- 存储 ----------
    database_path: str = "data/database/classroom.db"
    event_image_dir: str = "data/events"
    log_dir: str = "logs"
    data_retention_days: int = 30

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @field_validator('dashscope_base_url')
    @classmethod
    def _dashscope_endpoint(cls, value):
        url = urlsplit(value)
        if (url.scheme != 'https' or not (url.hostname or '').endswith('.aliyuncs.com')
                or url.username or url.password or url.query or url.fragment
                or url.path.rstrip('/') != '/compatible-mode/v1'):
            raise ValueError('DASHSCOPE_BASE_URL 必须是阿里云 HTTPS OpenAI兼容接口地址')
        return value.rstrip('/')

    @model_validator(mode='after')
    def _audio_window(self):
        if self.audio_pre_seconds + self.audio_post_seconds + 2 > self.audio_buffer_seconds:
            raise ValueError('音频缓存必须大于事件前后时间窗并预留2秒')
        return self

    # ---------- 派生属性 ----------
    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    @property
    def db_file(self) -> Path:
        return self._abs(self.database_path)

    @property
    def events_dir(self) -> Path:
        return self._abs(self.event_image_dir)

    @property
    def logs_dir(self) -> Path:
        return self._abs(self.log_dir)

    @property
    def model_file(self) -> Path:
        return self._abs(self.yolo_model_path)

    @property
    def frontend_dist(self) -> Path:
        return BASE_DIR / "frontend" / "dist"

    @property
    def camera_indices(self) -> list[int]:
        out: list[int] = []
        for chunk in self.camera_device_indices.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                out.append(int(chunk))
            except ValueError:
                continue
        return out

    @property
    def allowed_origins(self) -> list[str]:
        """CORS 与 WebSocket Origin 共用的来源白名单。

        默认只信任本机（前端开发服务器 + 后端自身）。WebSocket 不受同源策略与
        CORS 约束，必须在握手阶段用这份名单校验 Origin，否则任意网页都能连上
        本机服务读取实时快照。
        """
        origins = {
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            f"http://127.0.0.1:{self.app_port}",
            f"http://localhost:{self.app_port}",
        }
        if self.app_host not in {"127.0.0.1", "localhost", "0.0.0.0", ""}:
            origins.add(f"http://{self.app_host}:{self.app_port}")
        for chunk in self.extra_allowed_origins.split(","):
            chunk = chunk.strip().rstrip("/")
            if chunk:
                origins.add(chunk)
        return sorted(origins)

    @property
    def qwen_configured(self) -> bool:
        """仅表示密钥是否存在，绝不对外暴露密钥内容。"""
        return bool(self.dashscope_api_key.strip())

    def resolve_path(self, value: str) -> Path:
        """把相对项目根目录的路径转成绝对路径。"""
        p = Path(value)
        return p if p.is_absolute() else BASE_DIR / p

    # 内部沿用的短名
    _abs = resolve_path

    def ensure_dirs(self) -> None:
        for path in (self.db_file.parent, self.events_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
