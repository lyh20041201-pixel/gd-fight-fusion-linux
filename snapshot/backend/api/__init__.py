from fastapi import APIRouter

from . import alerts, cameras, devices, events, readings, rules, settings, system, ws

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(devices.router)
api_router.include_router(cameras.router)
api_router.include_router(events.router)
api_router.include_router(readings.router)
api_router.include_router(rules.router)
api_router.include_router(settings.router)
api_router.include_router(alerts.router)

ws_router = ws.router

__all__ = ["api_router", "ws_router"]
