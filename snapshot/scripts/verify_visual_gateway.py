"""Verify actual candidate service -> alert controller -> simulated gateway ACK."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import asyncio,json
from types import SimpleNamespace
import cv2
from backend.config.settings import Settings
from backend.config.runtime import RuntimeConfig
from backend.database.db import Database
from backend.database.repositories import Repository
from backend.devices.manager import DeviceManager
from backend.devices.simulator import SimulatedGateway
from backend.alerts.controller import AlertController
from backend.schemas.enums import SafetyState
from backend.services.visual_events import VisualEventService
from backend.vision.video_events import VideoPrediction
from backend.ai.qwen_reviewer import QwenResult

async def main():
    root=Path('results/video_events/airtlab_r3d18');out=root/'gateway_verification';out.mkdir(exist_ok=True)
    source=Database(root/'replay/events.db');source.connect();src=Repository(source)
    samples=json.loads((root/'replay_summary.json').read_text(encoding='utf-8'))['samples'][:3]
    db=Database(out/'events.db');db.init_schema();repo=Repository(db)
    if repo.count_events():raise RuntimeError('Use fresh gateway verification output')
    settings=Settings(_env_file=None,simulation_mode=True);runtime=RuntimeConfig(settings,repo);runtime.load()
    devices=DeviceManager(SimulatedGateway(loss_rate=0,seed=42),runtime);await devices.start()
    controller=AlertController(devices,runtime,repo);acks=[]
    async def alarm(e):
        ack=await controller.apply_safety_state(SafetyState.ALARM,'模拟视觉确认',event_id=e.event_id)
        acks.append(ack);return ack
    service=VisualEventService(repo,out/'evidence',SimpleNamespace(submit=lambda t:True),lambda *x:None,alarm)
    evidence=[]
    try:
        for i,s in enumerate(samples):
            previous=src.get_event(s['event_id']);v=previous.rule_basis['visual']
            frames=[(f.captured_at,cv2.imread(str(root/'replay/evidence'/f.raw_path)),{}) for f in previous.frames]
            eid=await service.create(VideoPrediction('suspected_fight',s['score'],s['start'],s['end'],v['model_version']),f'C{i}',frames)
            assert len(acks)==(1 if i else 0)
            res=QwenResult('timeout',error='SIMULATED timeout') if i==2 else QwenResult('ok',payload={
                'event_type':'suspected_fight','decision':'confirmed' if i==0 else 'rejected',
                'summary':'模拟复核，仅验证协议','evidence':[{'frame_index':3,'observation':'模拟证据'}]})
            await service.on_result(eid,res);await service.on_result(eid,res)
            evidence.append(repo.get_event(eid).model_dump(mode='json'))
        assert len(acks)==1 and acks[0].get('ok')
        assert devices.adapter.simulated
        (out/'summary.json').write_text(json.dumps(dict(mode='simulated_gateway_real_protocol',api_calls=0,
            ack_count=len(acks),acks=acks,events=evidence),ensure_ascii=False,indent=2),encoding='utf-8')
    finally:
        await devices.stop();source.close();db.close()

if __name__=='__main__':asyncio.run(main())
