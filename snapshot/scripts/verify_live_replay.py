"""Replay local validation video through live models, then persist isolated evidence.

Run using the web .venv. This never inserts replay events into the live database.
"""
from pathlib import Path
import sys, asyncio, json, time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import torch
from backend.services.live_actions import InferenceWorker, LiveActionService
from backend.services.visual_events import VisualEventService
from backend.database.db import Database
from backend.database.repositories import Repository
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/live_actions/runtime_validation/replay'
SAMPLE='f79cd02858173d60254e4d43'
SOURCE=ROOT/'datasets/fallvision/raw/Fall Detection Video Dataset/Fall/Chair/Raw Video/f_raw_c_2/f_raw_c_2/C_N_78_resized.mp4'

async def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'config/live_actions.json').read_text(encoding='utf-8'))
    cache=torch.load(ROOT/f'datasets/video_events/skeleton_rebuild/fallvision/{SAMPLE}.pt',map_location='cpu',weights_only=True)
    worker=InferenceWorker(manifest['runtime']['python'],ROOT/'config/live_actions.json',ROOT,OUT/'logs')
    db=Database(OUT/'replay.sqlite');db.init_schema();repo=Repository(db)
    messages=[]
    async def alarm(event): raise AssertionError('An unreviewed replay must not activate hardware')
    service=VisualEventService(repo,OUT/'events',None,lambda *args:messages.append(args),alarm)
    bridge=LiveActionService(SimpleNamespace(live_actions_enabled=True),None,service,lambda *args:None,False)
    report=dict(source='validation video replay, not a live staged fall',sample_id=SAMPLE,windows=[])
    try:
        report['runtime']=await asyncio.to_thread(worker.start)
        capture=cv2.VideoCapture(str(SOURCE));assert capture.isOpened()
        try:
            for index,clip in enumerate(cache['clips']):
                relative=clip['timestamps'];origin=time.time()-float(relative[-1])
                frames=[]
                for frame_index,offset in zip(clip['frame_indices'],relative):
                    capture.set(cv2.CAP_PROP_POS_FRAMES,int(frame_index));ok,im=capture.read()
                    assert ok,frame_index
                    frames.append((origin+float(offset),im,{}))
                result=await asyncio.to_thread(worker.predict,f'REPLAY-{index}',frames)
                report['windows'].append(result)
                print('WINDOW',index,json.dumps(result,ensure_ascii=False),flush=True)
                fall=next((a for a in result['actions'] if a['kind']=='fall'),{})
                if fall.get('state')!='candidate':continue
                await bridge.create_event(fall,'REPLAY-FALL',frames)
                total,events=repo.list_events(camera_id='REPLAY-FALL',limit=1)
                assert total and events
                event=repo.get_event(events[0].event_id)
                assert event.rule_basis['visual']['state']=='manual_review'
                assert len(event.frames)==8
                evidence=OUT/'events'/event.rule_basis['visual']['evidence_clip']
                check=cv2.VideoCapture(str(evidence));ok,_=check.read();check.release();assert ok
                repo.update_event(event.event_id,title='[验收回放] 疑似人员倒地',description='真实数据集视频回放；不是摄像头现场事件。待人工核查。')
                report.update(status='passed',event_id=event.event_id,evidence_frames=len(event.frames),
                    evidence_video=str(evidence),review_state='manual_review',hardware_alarm_requested=False,
                    database=str(db.path),broadcast_types=[args[0] for args in messages])
                break
        finally:capture.release()
        assert report.get('status')=='passed','Selected replay produced no fall candidate'
    finally:
        worker.close();db.close()
        (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('REPLAY PASSED',report['event_id'],flush=True)

if __name__=='__main__':asyncio.run(main())
