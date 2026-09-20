"""统一日志：控制台 + 轮转文件，并对疑似密钥做脱敏。"""

from __future__ import annotations

import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)(api[_-]?key\"?\s*[:=]\s*)([^\s,\"']+)"),
]

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"


class SecretRedactingFilter(logging.Filter):
    """确保 API 密钥不会出现在任何日志文件中。"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover
            return True
        redacted = message
        for pattern in _SECRET_PATTERNS:
            redacted = pattern.sub(
                lambda m: (m.group(1) + "***") if m.lastindex else "***", redacted
            )
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    # Windows 控制台默认 GBK，会把中文日志打成乱码
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover
                pass
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT)
    redactor = SecretRedactingFilter()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(redactor)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        log_dir / "backend.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(redactor)
    root.addHandler(file_handler)

    error_handler = RotatingFileHandler(
        log_dir / "error.log", maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)
    error_handler.addFilter(redactor)
    root.addHandler(error_handler)

    # 降低第三方库噪音
    for noisy in ("uvicorn.access", "httpx", "httpcore", "openai", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
