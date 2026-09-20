"""风险事件相关数据模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .enums import EventSource, EventStatus, EventType, RiskLevel


class EventFrameOut(BaseModel):
    frame_id: int | None = None
    event_id: str
    camera_id: str
    role: str  # pre / trigger / post
    offset_seconds: float = 0.0
    raw_path: str | None = None
    annotated_path: str | None = None
    captured_at: float


class QwenAnalysisOut(BaseModel):
    analysis_id: int | None = None
    event_id: str
    model: str
    status: str  # ok / failed / skipped / timeout
    event_type: str | None = None
    risk_level: str | None = None
    summary: str | None = None
    evidence: list[str] = Field(default_factory=list)
    related_track_ids: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    needs_human_review: bool | None = None
    audio_assessment: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost: float | None = None
    created_at: float


class OperatorActionOut(BaseModel):
    action_id: int | None = None
    event_id: str | None = None
    action: str
    operator: str = "local"
    note: str | None = None
    created_at: float


class RiskEventOut(BaseModel):
    event_id: str
    event_type: EventType | str
    event_type_label: str = ""
    risk_level: RiskLevel
    source: EventSource
    camera_id: str | None = None
    track_ids: list[str] = Field(default_factory=list)
    occurred_at: float
    status: EventStatus = EventStatus.PENDING
    title: str = ""
    description: str = ""
    raw_image: str | None = None
    annotated_image: str | None = None
    sensor_snapshot: dict[str, Any] = Field(default_factory=dict)
    track_summary: dict[str, Any] = Field(default_factory=dict)
    rule_basis: dict[str, Any] = Field(default_factory=dict)
    alert_result: dict[str, Any] = Field(default_factory=dict)
    note: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    frames: list[EventFrameOut] = Field(default_factory=list)
    qwen: QwenAnalysisOut | None = None
    operator_actions: list[OperatorActionOut] = Field(default_factory=list)


class EventPatch(BaseModel):
    status: EventStatus | None = None
    note: str | None = None
    operator: str = "local"


class EventListOut(BaseModel):
    total: int
    items: list[RiskEventOut]
