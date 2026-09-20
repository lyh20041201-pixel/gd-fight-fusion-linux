"""规则接口。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from ..schemas.api import RuleOut, RulePatch
from ..services.state import AppState
from .deps import get_state

router = APIRouter(prefix="/api/rules", tags=["rules"])


@router.get("", response_model=list[RuleOut])
async def list_rules(state: AppState = Depends(get_state)) -> list[RuleOut]:
    if state.rules is None:
        return []
    return [
        RuleOut(
            rule_id=r["rule_id"],
            name=r["name"],
            description=r.get("description", ""),
            enabled=bool(r.get("enabled", True)),
            event_type=r["event_type"],
            risk_level=r["risk_level"],
            params=r.get("params", {}),
            editable_params=r.get("editable_params", []),
            last_triggered_at=r.get("last_triggered_at"),
            state=r.get("state", "idle"),
        )
        for r in state.rules.rules()
    ]


@router.put("/{rule_id}", response_model=RuleOut)
async def update_rule(
    rule_id: str, payload: RulePatch, state: AppState = Depends(get_state)
) -> RuleOut:
    if state.rules is None:
        raise HTTPException(status_code=503, detail="规则引擎未就绪")
    rule = state.rules.rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="规则不存在")
    if payload.params:
        # 取值范围已由 RulePatch 校验，这里只管"这条规则是否允许改这个参数"
        illegal = sorted(set(payload.params) - set(rule.get("editable_params", [])))
        if illegal:
            raise HTTPException(
                status_code=400,
                detail=f"规则 {rule_id} 不支持修改参数：{'、'.join(illegal)}",
            )
    risk_level = payload.risk_level.value if payload.risk_level else None
    updated = state.rules.update_rule(
        rule_id, enabled=payload.enabled, risk_level=risk_level, params=payload.params
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="规则不存在")
    await asyncio.to_thread(
        state.repo.save_rule_state,
        rule_id,
        enabled=payload.enabled,
        risk_level=risk_level,
        params=updated.get("params"),
    )
    state.repo.insert_operator_action(None, f"rule_update:{rule_id}", "local", str(payload.params))
    rule_state = state.rules.state_of(rule_id)
    return RuleOut(
        rule_id=updated["rule_id"],
        name=updated["name"],
        description=updated.get("description", ""),
        enabled=bool(updated.get("enabled", True)),
        event_type=updated["event_type"],
        risk_level=updated["risk_level"],
        params=updated.get("params", {}),
        editable_params=updated.get("editable_params", []),
        last_triggered_at=rule_state.last_triggered_at if rule_state else None,
        state=rule_state.label if rule_state else "idle",
    )
