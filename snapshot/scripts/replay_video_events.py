"""Public test replay: real local predictions, explicitly simulated review outcomes."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import asyncio,json,time
import cv2
from backend.database.db import Database
from backend.database.repositories import Repository
from backend.services.visual_events import VisualEventService
from backend.vision.video_events import VideoEventDetector
from backend.ai.qwen_reviewer import QwenResult
from types import SimpleNamespace

async def main():
    root=Path('results/video_events/airtlab_r3d18');manifest=json.loads(Path('datasets/video_events/airtlab.json').read_text(encoding='utf-8'))
    detector=VideoEventDetector(str(root/'finetune_best.pt'))
    out=root/'replay';out.mkdir(exist_ok=True)
    db=Database(out/'events.db');db.init_schema();repo=Repository(db)
    if repo.count_events():
        raise RuntimeError('Replay output already contains events; use a fresh output directory')
    calls=[];pending=[];samples=[]
    async def alarm(e):calls.append(e.event_id);return {'ok':True,'simulated':True,'status':'mock_ack'}
    service=VisualEventService(repo,out/'evidence',SimpleNamespace(submit=lambda t:pending.append(t) or True),lambda *x:None,alarm)
    rows=[r for r in manifest['rows'] if r['split']=='test']
    for i,row in enumerate(rows):
        cap=cv2.VideoCapture(row['path']);fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));frames=[]
        for index in range(0,n,max(1,round(fps/5))):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,im=cap.read()
            if ok:frames.append((index/fps,im,{}))
        cap.release()
        for end in range(15,len(frames),5):
            window=frames[max(0,end-20):end+1]
            for pred in detector.predict(window):
                eid=await service.create(pred,f'REPLAY-{i:03}',window)
                if eid:samples.append(dict(event_id=eid,path=row['path'],truth=row['label'],start=pred.start,end=pred.end,score=pred.score))
        print('replay',i+1,len(rows),len(samples),flush=True)
    # No truth labels enter the mock reviewer. Exercise each state independently.
    for i,task in enumerate(pending):
        mode=i%3
        r=QwenResult('timeout',error='SIMULATED timeout') if mode==2 else QwenResult('ok',payload={
            'event_type':'suspected_fight','decision':'confirmed' if mode==0 else 'rejected',
            'summary':'模拟协议测试，非真实多模态判断','evidence':[{'frame_index':3,'observation':'模拟输出仅用于状态机验收'}]})
        await service.on_result(task.event_id,r);await service.on_result(task.event_id,r)
    result=dict(local_inference='real R3D18',multimodal='SIMULATED, no API calls',
        samples=samples,candidate_count=len(samples),alarm_count=len(calls),
        expected_mock_alarm_count=sum(i%3==0 for i in range(len(pending))),
        states={s:repo.db.query_one("SELECT count(*) c FROM risk_events WHERE json_extract(rule_basis,'$.visual.state')=?",(s,))['c']
                for s in ['confirmed','rejected','manual_review']})
    assert result['alarm_count']==result['expected_mock_alarm_count']
    (root/'replay_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'multimodal_review_manifest.json').write_text(json.dumps([dict(**s,frames=8) for s in samples],ensure_ascii=False,indent=2),encoding='utf-8')
    db.close()

if __name__=='__main__':asyncio.run(main())
