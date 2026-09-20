"""pytest 公共夹具。

所有测试都在模拟模式下运行，使用临时数据库与临时事件目录，
不会触碰真实串口、真实摄像头或 QWEN。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def temp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMULATION_MODE", "true")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EVENT_IMAGE_DIR", str(tmp_path / "events"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    monkeypatch.setenv("SIMULATED_CAMERA_COUNT", "2")
    from backend.config.settings import get_settings

    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture()
def settings(temp_env):
    from backend.config.settings import get_settings

    return get_settings()


@pytest.fixture()
def repo(settings):
    from backend.database.db import Database
    from backend.database.repositories import Repository

    db = Database(settings.db_file)
    db.init_schema()
    yield Repository(db)
    db.close()


@pytest.fixture()
def runtime(settings, repo):
    from backend.config.runtime import RuntimeConfig

    config = RuntimeConfig(settings, repo)
    config.load()
    return config
