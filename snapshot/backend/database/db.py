"""SQLite 连接管理。

- 开启 WAL，适合本地单进程 + 多线程读写。
- 所有异常收敛为 DatabaseError，不允许因为数据库问题让采集/报警线程崩溃。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

logger = logging.getLogger(__name__)

SCHEMA_FILE = Path(__file__).with_name("schema.sql")


class DatabaseError(RuntimeError):
    """数据库访问异常（已被捕获并转换，调用方可安全降级）。"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._healthy = False
        self._last_error: str | None = None

    # ---------- 生命周期 ----------
    def connect(self) -> None:
        with self._lock:
            if self._conn is not None:
                return
            try:
                conn = sqlite3.connect(
                    str(self.path), check_same_thread=False, timeout=10.0
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("PRAGMA busy_timeout=5000")
                self._conn = conn
                self._healthy = True
                self._last_error = None
            except sqlite3.Error as exc:  # pragma: no cover - 环境相关
                self._healthy = False
                self._last_error = str(exc)
                raise DatabaseError(f"无法打开数据库 {self.path}: {exc}") from exc

    def init_schema(self) -> None:
        self.connect()
        sql = SCHEMA_FILE.read_text(encoding="utf-8")
        with self._lock:
            assert self._conn is not None
            try:
                self._conn.executescript(sql)
                self._conn.commit()
            except sqlite3.Error as exc:
                self._healthy = False
                self._last_error = str(exc)
                raise DatabaseError(f"初始化数据库结构失败: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.commit()
                    self._conn.close()
                finally:
                    self._conn = None

    # ---------- 健康 ----------
    @property
    def healthy(self) -> bool:
        return self._healthy

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def check(self) -> bool:
        try:
            self.query_one("SELECT 1 AS ok")
            self._healthy = True
            self._last_error = None
        except DatabaseError as exc:
            self._healthy = False
            self._last_error = str(exc)
        return self._healthy

    # ---------- 基础操作 ----------
    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        self.connect()
        with self._lock:
            assert self._conn is not None
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except sqlite3.Error as exc:
                self._conn.rollback()
                self._healthy = False
                self._last_error = str(exc)
                logger.error("SQLite 操作失败: %s", exc)
                raise DatabaseError(str(exc)) from exc
            finally:
                cur.close()

    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> int:
        """执行写语句，返回新插入行的 rowid。

        仅对 INSERT 有意义：sqlite3 的 lastrowid 在 DELETE/UPDATE 上会残留连接上
        一次插入的 rowid，因此受影响行数请改用 execute_write。
        """
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.lastrowid or cur.rowcount

    def execute_write(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> int:
        """执行 DELETE / UPDATE，返回真实受影响行数。"""
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount

    def executemany(self, sql: str, seq: Sequence[Sequence[Any]]) -> None:
        with self.cursor() as cur:
            cur.executemany(sql, seq)

    def query_all(
        self, sql: str, params: Sequence[Any] | dict[str, Any] = ()
    ) -> list[sqlite3.Row]:
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def query_one(
        self, sql: str, params: Sequence[Any] | dict[str, Any] = ()
    ) -> sqlite3.Row | None:
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()


# ---------- JSON 辅助 ----------
def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def now() -> float:
    return time.time()
