"""数据访问层。

所有方法都是同步的（本地 SQLite 足够快）；在事件循环中调用的写操作
由调用方通过 asyncio.to_thread 包装，避免阻塞。
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any, Iterable

from ..schemas.enums import EVENT_TYPE_LABELS
from ..schemas.events import (
    EventFrameOut,
    QwenAnalysisOut,
    OperatorActionOut,
    RiskEventOut,
)
from .db import Database, DatabaseError, dumps, loads

logger = logging.getLogger(__name__)


class Repository:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ================= 设备 =================
    def upsert_device(
        self,
        device_id: str,
        name: str,
        device_type: str,
        *,
        capabilities: Iterable[str] = (),
        meta: dict[str, Any] | None = None,
    ) -> None:
        ts = time.time()
        self.db.execute(
            """
            INSERT INTO devices (device_id, name, device_type, capabilities, meta,
                                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(device_id) DO UPDATE SET
                name = excluded.name,
                device_type = excluded.device_type,
                capabilities = excluded.capabilities,
                updated_at = excluded.updated_at
            """,
            (
                device_id,
                name,
                device_type,
                dumps(list(capabilities)),
                dumps(meta or {}),
                ts,
                ts,
            ),
        )

    def update_device_status(
        self,
        device_id: str,
        *,
        online: bool,
        state: str,
        rssi: int | None,
        packet_rate: float | None,
        last_seen: float | None,
        last_heartbeat: float | None,
        fault_reason: str | None,
    ) -> None:
        self.db.execute(
            """
            UPDATE devices SET online = ?, state = ?, rssi = ?, packet_rate = ?,
                   last_seen = ?, last_heartbeat = ?, fault_reason = ?, updated_at = ?
            WHERE device_id = ?
            """,
            (
                int(online),
                state,
                rssi,
                packet_rate,
                last_seen,
                last_heartbeat,
                fault_reason,
                time.time(),
                device_id,
            ),
        )

    def list_devices(self) -> list[dict[str, Any]]:
        rows = self.db.query_all("SELECT * FROM devices ORDER BY device_id")
        return [dict(r) for r in rows]

    # ================= 传感器 =================
    def upsert_sensor(
        self, sensor_id: str, device_id: str, kind: str, name: str, unit: str
    ) -> None:
        self.db.execute(
            """
            INSERT INTO sensors (sensor_id, device_id, sensor_kind, name, unit, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sensor_id) DO UPDATE SET
                name = excluded.name, unit = excluded.unit, device_id = excluded.device_id
            """,
            (sensor_id, device_id, kind, name, unit, time.time()),
        )

    def insert_readings(self, rows: list[tuple[str, str, str, float, str, float]]) -> None:
        if not rows:
            return
        self.db.executemany(
            """
            INSERT INTO sensor_readings
                (sensor_id, device_id, sensor_kind, value, unit, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def latest_reading(self, sensor_kind: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM sensor_readings WHERE sensor_kind = ? "
            "ORDER BY recorded_at DESC LIMIT 1",
            (sensor_kind,),
        )
        return dict(row) if row else None

    def reading_series(
        self, sensor_kind: str, start: float, end: float, bucket: int
    ) -> list[list[float]]:
        rows = self.db.query_all(
            """
            SELECT CAST(recorded_at / ? AS INTEGER) * ? AS bucket_ts,
                   AVG(value) AS v
            FROM sensor_readings
            WHERE sensor_kind = ? AND recorded_at BETWEEN ? AND ?
            GROUP BY bucket_ts ORDER BY bucket_ts
            """,
            (bucket, bucket, sensor_kind, start, end),
        )
        return [[float(r["bucket_ts"]), round(float(r["v"]), 2)] for r in rows]

    # ================= 摄像头 =================
    def upsert_camera(
        self,
        camera_id: str,
        name: str,
        region: str,
        source: str,
        roi: list[list[float]],
        is_primary_overlap: bool = True,
        enabled: bool = True,
    ) -> None:
        ts = time.time()
        self.db.execute(
            """
            INSERT INTO cameras (camera_id, name, region, source, enabled, roi,
                                 is_primary_overlap, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(camera_id) DO UPDATE SET
                source = excluded.source, updated_at = excluded.updated_at
            """,
            (
                camera_id,
                name,
                region,
                source,
                int(enabled),
                dumps(roi),
                int(is_primary_overlap),
                ts,
                ts,
            ),
        )

    def update_camera_config(self, camera_id: str, **fields: Any) -> None:
        allowed = {"name", "region", "roi", "is_primary_overlap", "enabled"}
        sets, params = [], []
        for key, value in fields.items():
            if key not in allowed or value is None:
                continue
            if key == "roi":
                value = dumps(value)
            if key in {"is_primary_overlap", "enabled"}:
                value = int(bool(value))
            sets.append(f"{key} = ?")
            params.append(value)
        if not sets:
            return
        sets.append("updated_at = ?")
        params.extend([time.time(), camera_id])
        self.db.execute(
            f"UPDATE cameras SET {', '.join(sets)} WHERE camera_id = ?", params
        )

    def list_cameras(self) -> list[dict[str, Any]]:
        rows = self.db.query_all("SELECT * FROM cameras ORDER BY camera_id")
        out = []
        for r in rows:
            d = dict(r)
            d["roi"] = loads(d.get("roi"), [])
            out.append(d)
        return out

    # ================= 人数 =================
    def insert_occupancy(self, camera_id: str | None, count: int, ts: float) -> None:
        self.db.execute(
            "INSERT INTO occupancy_samples (camera_id, person_count, recorded_at) "
            "VALUES (?, ?, ?)",
            (camera_id, count, ts),
        )

    def occupancy_series(
        self, start: float, end: float, bucket: int, camera_id: str | None = None
    ) -> list[list[float]]:
        if camera_id:
            where, params = "camera_id = ?", [camera_id]
        else:
            where, params = "camera_id IS NULL", []
        rows = self.db.query_all(
            f"""
            SELECT CAST(recorded_at / ? AS INTEGER) * ? AS bucket_ts,
                   AVG(person_count) AS v
            FROM occupancy_samples
            WHERE {where} AND recorded_at BETWEEN ? AND ?
            GROUP BY bucket_ts ORDER BY bucket_ts
            """,
            [bucket, bucket, *params, start, end],
        )
        return [[float(r["bucket_ts"]), round(float(r["v"]), 1)] for r in rows]

    # ================= 风险事件 =================
    def insert_event(self, event: RiskEventOut) -> None:
        self.db.execute(
            """
            INSERT INTO risk_events
                (event_id, event_type, risk_level, source, camera_id, track_ids,
                 occurred_at, status, title, description, raw_image, annotated_image,
                 sensor_snapshot, track_summary, rule_basis, alert_result, note,
                 created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                event.event_id,
                str(getattr(event.event_type, "value", event.event_type)),
                event.risk_level.value,
                event.source.value,
                event.camera_id,
                dumps(event.track_ids),
                event.occurred_at,
                event.status.value,
                event.title,
                event.description,
                event.raw_image,
                event.annotated_image,
                dumps(event.sensor_snapshot),
                dumps(event.track_summary),
                dumps(event.rule_basis),
                dumps(event.alert_result),
                event.note,
                event.created_at,
                event.updated_at,
            ),
        )

    def update_event(self, event_id: str, **fields: Any) -> None:
        allowed = {
            "rule_basis",
            "status",
            "note",
            "raw_image",
            "annotated_image",
            "alert_result",
            "risk_level",
            "description",
            "track_summary",
        }
        sets, params = [], []
        for key, value in fields.items():
            if key not in allowed or value is None:
                continue
            if isinstance(value, (dict, list)):
                value = dumps(value)
            sets.append(f"{key} = ?")
            params.append(value)
        if not sets:
            return
        sets.append("updated_at = ?")
        params.extend([time.time(), event_id])
        self.db.execute(
            f"UPDATE risk_events SET {', '.join(sets)} WHERE event_id = ?", params
        )

    def _row_to_event(self, row: sqlite3.Row) -> RiskEventOut:
        d = dict(row)
        event_type = d["event_type"]
        return RiskEventOut(
            event_id=d["event_id"],
            event_type=event_type,
            event_type_label=EVENT_TYPE_LABELS.get(event_type, event_type),
            risk_level=d["risk_level"],
            source=d["source"],
            camera_id=d["camera_id"],
            track_ids=loads(d["track_ids"], []),
            occurred_at=d["occurred_at"],
            status=d["status"],
            title=d["title"] or "",
            description=d["description"] or "",
            raw_image=d["raw_image"],
            annotated_image=d["annotated_image"],
            sensor_snapshot=loads(d["sensor_snapshot"], {}),
            track_summary=loads(d["track_summary"], {}),
            rule_basis=loads(d["rule_basis"], {}),
            alert_result=loads(d["alert_result"], {}),
            note=d["note"],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
        )

    def list_events(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        risk_level: str | None = None,
        event_type: str | None = None,
        camera_id: str | None = None,
        start: float | None = None,
        end: float | None = None,
    ) -> tuple[int, list[RiskEventOut]]:
        where, params = [], []
        if status:
            where.append("status = ?")
            params.append(status)
        if risk_level:
            where.append("risk_level = ?")
            params.append(risk_level)
        if event_type:
            where.append("event_type = ?")
            params.append(event_type)
        if camera_id:
            where.append("camera_id = ?")
            params.append(camera_id)
        if start is not None:
            where.append("occurred_at >= ?")
            params.append(start)
        if end is not None:
            where.append("occurred_at <= ?")
            params.append(end)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        total_row = self.db.query_one(
            f"SELECT COUNT(*) AS c FROM risk_events {clause}", params
        )
        total = int(total_row["c"]) if total_row else 0
        rows = self.db.query_all(
            f"SELECT * FROM risk_events {clause} ORDER BY occurred_at DESC "
            f"LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        return total, [self._row_to_event(r) for r in rows]

    def get_event(self, event_id: str, *, with_children: bool = True) -> RiskEventOut | None:
        row = self.db.query_one(
            "SELECT * FROM risk_events WHERE event_id = ?", (event_id,)
        )
        if row is None:
            return None
        event = self._row_to_event(row)
        if with_children:
            event.frames = self.list_frames(event_id)
            event.qwen = self.latest_qwen(event_id)
            event.operator_actions = self.list_operator_actions(event_id)
        return event

    def count_events(self, *, status: str | None = None, since: float | None = None) -> int:
        where, params = [], []
        if status:
            where.append("status = ?")
            params.append(status)
        if since is not None:
            where.append("occurred_at >= ?")
            params.append(since)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        row = self.db.query_one(f"SELECT COUNT(*) AS c FROM risk_events {clause}", params)
        return int(row["c"]) if row else 0

    def event_count_series(self, start: float, end: float, bucket: int) -> list[list[float]]:
        rows = self.db.query_all(
            """
            SELECT CAST(occurred_at / ? AS INTEGER) * ? AS bucket_ts, COUNT(*) AS c
            FROM risk_events WHERE occurred_at BETWEEN ? AND ?
            GROUP BY bucket_ts ORDER BY bucket_ts
            """,
            (bucket, bucket, start, end),
        )
        return [[float(r["bucket_ts"]), float(r["c"])] for r in rows]

    # ================= 事件关键帧 =================
    def insert_frame(self, frame: EventFrameOut) -> int:
        return self.db.execute(
            """
            INSERT INTO event_frames
                (event_id, camera_id, role, offset_seconds, raw_path, annotated_path,
                 captured_at)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                frame.event_id,
                frame.camera_id,
                frame.role,
                frame.offset_seconds,
                frame.raw_path,
                frame.annotated_path,
                frame.captured_at,
            ),
        )

    def list_frames(self, event_id: str) -> list[EventFrameOut]:
        rows = self.db.query_all(
            "SELECT * FROM event_frames WHERE event_id = ? ORDER BY captured_at",
            (event_id,),
        )
        return [
            EventFrameOut(
                frame_id=r["id"],
                event_id=r["event_id"],
                camera_id=r["camera_id"],
                role=r["role"],
                offset_seconds=r["offset_seconds"],
                raw_path=r["raw_path"],
                annotated_path=r["annotated_path"],
                captured_at=r["captured_at"],
            )
            for r in rows
        ]

    # ================= Qwen =================
    def insert_qwen(self, analysis: QwenAnalysisOut, raw_response: str | None = None) -> int:
        return self.db.execute(
            """
            INSERT INTO gpt_analyses
                (event_id, model, status, event_type, risk_level, summary, evidence,
                 related_track_ids, recommended_actions, needs_human_review, raw_response,
                 error, latency_ms, prompt_tokens, completion_tokens, estimated_cost,
                 created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                analysis.event_id,
                analysis.model,
                analysis.status,
                analysis.event_type,
                analysis.risk_level,
                analysis.summary,
                dumps(analysis.evidence),
                dumps(analysis.related_track_ids),
                dumps(analysis.recommended_actions),
                None if analysis.needs_human_review is None else int(analysis.needs_human_review),
                raw_response,
                analysis.error,
                analysis.latency_ms,
                analysis.prompt_tokens,
                analysis.completion_tokens,
                analysis.estimated_cost,
                analysis.created_at,
            ),
        )

    def latest_qwen(self, event_id: str) -> QwenAnalysisOut | None:
        row = self.db.query_one(
            "SELECT * FROM gpt_analyses WHERE event_id = ? ORDER BY created_at DESC LIMIT 1",
            (event_id,),
        )
        if not row:
            return None
        d = dict(row)
        return QwenAnalysisOut(
            analysis_id=d["id"],
            event_id=d["event_id"],
            model=d["model"],
            status=d["status"],
            event_type=d["event_type"],
            risk_level=d["risk_level"],
            summary=d["summary"],
            evidence=loads(d["evidence"], []),
            related_track_ids=loads(d["related_track_ids"], []),
            recommended_actions=loads(d["recommended_actions"], []),
            needs_human_review=(
                None if d["needs_human_review"] is None else bool(d["needs_human_review"])
            ),
            error=d["error"],
            latency_ms=d["latency_ms"],
            prompt_tokens=d["prompt_tokens"],
            completion_tokens=d["completion_tokens"],
            audio_assessment=(loads(d['raw_response'], {}) or {}).get('audio_assessment') if d['status'] == 'ok' else None,
            estimated_cost=d["estimated_cost"],
            created_at=d["created_at"],
        )

    def qwen_usage(self) -> dict[str, Any]:
        row = self.db.query_one(
            """
            SELECT COUNT(*) AS calls,
                   COALESCE(SUM(prompt_tokens), 0) AS pt,
                   COALESCE(SUM(completion_tokens), 0) AS ct,
                   COALESCE(SUM(estimated_cost), 0) AS cost,
                   SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END) AS failures
            FROM gpt_analyses
            WHERE model IN ('qwen3.5-omni-flash', 'qwen3.5-omni-plus')
            """
        )
        if not row:
            return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
        return {
            "calls": int(row["calls"]),
            "prompt_tokens": int(row["pt"]),
            "completion_tokens": int(row["ct"]),
            "estimated_cost": round(float(row["cost"]), 4),
            "failures": int(row["failures"] or 0),
        }

    # ================= 报警与操作日志 =================
    def insert_alert_action(
        self,
        *,
        event_id: str | None,
        request_id: str | None,
        target: str,
        action: str,
        payload: dict[str, Any],
        status: str,
        simulated: bool,
        retries: int = 0,
        acked_at: float | None = None,
    ) -> int:
        return self.db.execute(
            """
            INSERT INTO alert_actions
                (event_id, request_id, target, action, payload, status, simulated,
                 retries, created_at, acked_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                event_id,
                request_id,
                target,
                action,
                dumps(payload),
                status,
                int(simulated),
                retries,
                time.time(),
                acked_at,
            ),
        )

    def update_alert_action(self, request_id: str, status: str, acked_at: float | None) -> None:
        self.db.execute(
            "UPDATE alert_actions SET status = ?, acked_at = ? WHERE request_id = ?",
            (status, acked_at, request_id),
        )

    def list_alert_actions(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.query_all(
            "SELECT * FROM alert_actions ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = loads(d.get("payload"), {})
            d["simulated"] = bool(d.get("simulated"))
            out.append(d)
        return out

    def insert_operator_action(
        self, event_id: str | None, action: str, operator: str, note: str | None
    ) -> int:
        return self.db.execute(
            "INSERT INTO operator_actions (event_id, action, operator, note, created_at) "
            "VALUES (?,?,?,?,?)",
            (event_id, action, operator, note, time.time()),
        )

    def list_operator_actions(self, event_id: str | None = None, limit: int = 100
                              ) -> list[OperatorActionOut]:
        if event_id:
            rows = self.db.query_all(
                "SELECT * FROM operator_actions WHERE event_id = ? ORDER BY created_at",
                (event_id,),
            )
        else:
            rows = self.db.query_all(
                "SELECT * FROM operator_actions ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        return [
            OperatorActionOut(
                action_id=r["id"],
                event_id=r["event_id"],
                action=r["action"],
                operator=r["operator"],
                note=r["note"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ================= 规则与设置 =================
    def upsert_rule(self, rule: dict[str, Any]) -> None:
        self.db.execute(
            """
            INSERT INTO rules (rule_id, name, description, enabled, event_type,
                               risk_level, params, updated_at)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(rule_id) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                updated_at = excluded.updated_at
            """,
            (
                rule["rule_id"],
                rule["name"],
                rule.get("description", ""),
                int(rule.get("enabled", True)),
                rule["event_type"],
                rule["risk_level"],
                dumps(rule.get("params", {})),
                time.time(),
            ),
        )

    def save_rule_state(
        self,
        rule_id: str,
        *,
        enabled: bool | None = None,
        risk_level: str | None = None,
        params: dict[str, Any] | None = None,
        last_triggered_at: float | None = None,
    ) -> None:
        sets, values = [], []
        if enabled is not None:
            sets.append("enabled = ?")
            values.append(int(enabled))
        if risk_level is not None:
            sets.append("risk_level = ?")
            values.append(risk_level)
        if params is not None:
            sets.append("params = ?")
            values.append(dumps(params))
        if last_triggered_at is not None:
            sets.append("last_triggered_at = ?")
            values.append(last_triggered_at)
        if not sets:
            return
        sets.append("updated_at = ?")
        values.extend([time.time(), rule_id])
        self.db.execute(f"UPDATE rules SET {', '.join(sets)} WHERE rule_id = ?", values)

    def list_rules(self) -> list[dict[str, Any]]:
        rows = self.db.query_all("SELECT * FROM rules ORDER BY rule_id")
        out = []
        for r in rows:
            d = dict(r)
            d["params"] = loads(d.get("params"), {})
            d["enabled"] = bool(d.get("enabled"))
            out.append(d)
        return out

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.db.query_one("SELECT value FROM settings WHERE key = ?", (key,))
        if not row:
            return default
        return loads(row["value"], default)

    def set_setting(self, key: str, value: Any) -> None:
        self.db.execute(
            """
            INSERT INTO settings (key, value, updated_at) VALUES (?,?,?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                          updated_at = excluded.updated_at
            """,
            (key, dumps(value), time.time()),
        )

    def all_settings(self) -> dict[str, Any]:
        rows = self.db.query_all("SELECT key, value FROM settings")
        return {r["key"]: loads(r["value"]) for r in rows}

    # ================= 日志 =================
    def insert_log(
        self, level: str, source: str, message: str, detail: dict[str, Any] | None = None
    ) -> None:
        try:
            self.db.execute(
                "INSERT INTO system_logs (level, source, message, detail, created_at) "
                "VALUES (?,?,?,?,?)",
                (level, source, message, dumps(detail or {}), time.time()),
            )
        except DatabaseError:  # 日志写库失败不得影响主流程
            logger.warning("系统日志写入数据库失败: %s", message)

    def list_logs(self, limit: int = 200, level: str | None = None) -> list[dict[str, Any]]:
        if level:
            rows = self.db.query_all(
                "SELECT * FROM system_logs WHERE level = ? ORDER BY created_at DESC LIMIT ?",
                (level, limit),
            )
        else:
            rows = self.db.query_all(
                "SELECT * FROM system_logs ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        out = []
        for r in rows:
            d = dict(r)
            d["detail"] = loads(d.get("detail"), {})
            out.append(d)
        return out

    # ================= 维护 =================
    def purge_old_data(self, retention_days: int) -> tuple[dict[str, int], list[str]]:
        """按保留期清理历史数据。

        返回 (各表删除条数, 需要一并删除的事件图片相对路径)。图片由调用方删除，
        数据库层不碰文件系统。event_frames 与 gpt_analyses 通过外键级联删除；
        alert_actions 与 operator_actions 没有外键，必须按时间显式清理，
        否则会连同事件图片一起无限增长。
        """
        # 与接口层的校验双保险：非正数会把 cutoff 推到未来，删光整张表。
        retention_days = max(1, int(retention_days))
        cutoff = time.time() - retention_days * 86400

        images = self._expired_event_images(cutoff)

        removed = {}
        removed["sensor_readings"] = self.db.execute_write(
            "DELETE FROM sensor_readings WHERE recorded_at < ?", (cutoff,)
        )
        removed["occupancy_samples"] = self.db.execute_write(
            "DELETE FROM occupancy_samples WHERE recorded_at < ?", (cutoff,)
        )
        removed["system_logs"] = self.db.execute_write(
            "DELETE FROM system_logs WHERE created_at < ?", (cutoff,)
        )
        removed["event_frames"] = self.db.execute_write(
            "DELETE FROM event_frames WHERE event_id IN "
            "(SELECT event_id FROM risk_events WHERE occurred_at < ?)",
            (cutoff,),
        )
        removed["gpt_analyses"] = self.db.execute_write(
            "DELETE FROM gpt_analyses WHERE event_id IN "
            "(SELECT event_id FROM risk_events WHERE occurred_at < ?)",
            (cutoff,),
        )
        removed["risk_events"] = self.db.execute_write(
            "DELETE FROM risk_events WHERE occurred_at < ?", (cutoff,)
        )
        removed["alert_actions"] = self.db.execute_write(
            "DELETE FROM alert_actions WHERE created_at < ?", (cutoff,)
        )
        removed["operator_actions"] = self.db.execute_write(
            "DELETE FROM operator_actions WHERE created_at < ?", (cutoff,)
        )
        return removed, images

    def _expired_event_images(self, cutoff: float) -> list[str]:
        """收集过期事件引用的全部图片（事件封面 + 关键帧的原图与标注图）。"""
        paths: set[str] = set()
        rows = self.db.query_all(
            "SELECT raw_image, annotated_image, rule_basis FROM risk_events WHERE occurred_at < ?",
            (cutoff,),
        )
        for row in rows:
            paths.update(p for p in (row["raw_image"], row["annotated_image"]) if p)
            basis = loads(row['rule_basis'], {})
            paths.update(p for p in ((basis.get('audio') or {}).get('path'),
                                      (basis.get('visual') or {}).get('evidence_clip')) if isinstance(p, str) and p)
        rows = self.db.query_all(
            "SELECT raw_path, annotated_path FROM event_frames WHERE event_id IN "
            "(SELECT event_id FROM risk_events WHERE occurred_at < ?)",
            (cutoff,),
        )
        for row in rows:
            paths.update(p for p in (row["raw_path"], row["annotated_path"]) if p)
        return sorted(paths)
