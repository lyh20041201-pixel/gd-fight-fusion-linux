"""Qwen 复核提示词与结构化输出 Schema。

原则：
- Qwen 只做"解释与复核"，绝不作为安全报警的唯一判断来源；
- 只提供已有关键帧、传感器 JSON、跟踪摘要与本地规则依据；
- 明确禁止 Qwen 生成任何检测框坐标。
"""

from __future__ import annotations

import json
from typing import Any

RISK_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "event_type",
        "risk_level",
        "summary",
        "evidence",
        "related_track_ids",
        "recommended_actions",
        "needs_human_review",
    ],
    "properties": {
        "event_type": {"type": "string"},
        "risk_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "summary": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "related_track_ids": {"type": "array", "items": {"type": "string"}},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
        "needs_human_review": {"type": "boolean"},
    },
}

SYSTEM_PROMPT = """你是教室环境安全监控系统的复核助手。
你会收到：若干张已由本地模型标注/未标注的关键帧、当前传感器读数、摄像头摘要、
人数、匿名 Track 摘要、本地规则触发依据与当前时间。

要求：
1. 只根据提供的材料判断，不要编造画面中不存在的内容。
2. 绝对不要输出任何检测框坐标；检测框只能由本地 YOLO / ByteTrack 产生。
3. 不做人脸识别，不推测任何人的身份、性别、年龄或个人特征。
4. 你的结论是"复核意见"，本地规则（尤其是烟感等硬报警）不会因你的判断被取消。
5. 若材料不足以判断，把 needs_human_review 设为 true，并在 summary 中说明缺少什么。
6. 使用简体中文输出，summary 控制在 60 字以内，evidence 每条 30 字以内。
"""


def build_user_payload(context: dict[str, Any]) -> str:
    """把事件上下文序列化成给模型的文本部分。"""
    lines = [
        f"当前时间: {context.get('local_time')}",
        f"教室安全状态: {context.get('safety_state')}",
        f"本地判定事件类型: {context.get('event_type')}（{context.get('event_type_label')}）",
        f"本地风险等级: {context.get('risk_level')}",
        f"本地规则依据: {json.dumps(context.get('rule_basis', {}), ensure_ascii=False)}",
        f"当前人数(融合后): {context.get('occupancy')}",
        f"各摄像头人数: {json.dumps(context.get('per_camera', {}), ensure_ascii=False)}",
        f"传感器读数: {json.dumps(context.get('sensors', {}), ensure_ascii=False)}",
        f"跟踪摘要: {json.dumps(context.get('track_summary', {}), ensure_ascii=False)}",
        f"摄像头摘要: {json.dumps(context.get('camera_summary', {}), ensure_ascii=False)}",
        "",
        "请复核该事件是否成立、风险等级是否合适，并给出处理建议。",
    ]
    return "\n".join(lines)


def validate_review(payload: dict[str, Any]) -> tuple[bool, str]:
    """轻量 JSON Schema 校验（不引入额外依赖）。"""
    if not isinstance(payload, dict) or set(payload) != set(RISK_REVIEW_SCHEMA['required']):
        return False, '复核字段不完整或存在额外字段'
    if not isinstance(payload['event_type'], str) or not payload['event_type'].strip():
        return False, '事件类型不能为空'
    for key in RISK_REVIEW_SCHEMA["required"]:
        if key not in payload:
            return False, f"缺少字段 {key}"
    if not isinstance(payload['risk_level'], str) or payload["risk_level"] not in {"low", "medium", "high", "critical"}:
        return False, f"risk_level 非法: {payload['risk_level']}"
    for key in ("evidence", "related_track_ids", "recommended_actions"):
        if not isinstance(payload[key], list):
            return False, f"{key} 必须是数组"
        if any(not isinstance(item, str) for item in payload[key]):
            return False, f"{key} 元素必须是字符串"
    if not isinstance(payload["needs_human_review"], bool):
        return False, "needs_human_review 必须是布尔值"
    if not isinstance(payload["summary"], str) or not payload["summary"].strip():
        return False, "summary 不能为空"
    return True, ""
