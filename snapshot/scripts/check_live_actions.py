"""Read-only acceptance snapshot of the running local camera/action service."""
import json,time,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/live_actions/runtime_validation'

def get(path):
    with urllib.request.urlopen('http://127.0.0.1:8000/api/'+path,timeout=10) as response:
        return json.load(response)

def main():
    samples=[]
    for index in range(8):
        cameras=get('cameras');actions=get('actions/status')
        samples.append(dict(at=time.time(),cameras=[dict(camera_id=c['camera_id'],online=c['online'],fps=c['fps']) for c in cameras],actions=actions))
        print(index,actions['state'],[(c['camera_id'],c['fps']) for c in cameras],flush=True)
        if index<7:time.sleep(2)
    last=samples[-3:]
    assert all(s['actions']['state']=='ready' for s in last)
    assert all(any(c['online'] and c['fps']>=10 for c in s['cameras']) for s in last)
    assert all(s['actions']['cameras']['CAM-01']['state']=='running' for s in last)
    assert len({s['actions']['cameras']['CAM-01']['updated_at'] for s in last})>=2
    assert all(a['state'] in ('normal','unknown','candidate') for s in last for a in s['actions']['cameras']['CAM-01']['actions'])
    result=dict(status='passed',health=get('health'),samples=samples,
        note='Real USB camera continuity and service health only; does not measure action accuracy.')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'live_camera_acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('LIVE CAMERA PASSED',flush=True)

if __name__=='__main__':main()
