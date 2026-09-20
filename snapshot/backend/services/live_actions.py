"""Bounded asynchronous camera inference and explicit live model health."""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid
from multiprocessing.connection import Client
from ..vision.video_events import VideoPrediction


class InferenceWorker:
    def __init__(self,python,manifest,root,log_dir):
        self.python,self.manifest,self.root=python,manifest,root
        self.log_dir=Path(log_dir);self.process=None;self.connection=None
        self.lock=threading.Lock()

    def start(self):
        if not Path(self.python).is_file():raise ValueError('动作推理Python环境不存在')
        if not Path(self.manifest).is_file():raise ValueError('动作识别部署清单不存在')
        pipe='\\\\.\\pipe\\gd-actions-'+uuid.uuid4().hex
        auth=os.urandom(32);env=os.environ.copy();env['GD_ACTION_AUTH']=auth.hex()
        env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        self.log_dir.mkdir(parents=True,exist_ok=True)
        with (self.log_dir/'live_actions_worker.log').open('ab') as log:
            self.process=subprocess.Popen([self.python,'-u',str(Path(self.root)/'scripts/serve_live_actions.py'),
                '--pipe',pipe,'--manifest',str(self.manifest)],cwd=self.root,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            if self.process.poll() is not None:raise RuntimeError('动作推理进程启动失败，请查看服务日志')
            try:
                self.connection=Client(pipe,family='AF_PIPE',authkey=auth)
                if not self.connection.poll(10):raise TimeoutError('动作模型初始化超时')
                return self.connection.recv()
            except FileNotFoundError:time.sleep(.1)
        raise TimeoutError('动作推理进程未就绪')

    def predict(self,camera,frames):
        return self._request(dict(camera=camera,frames=[(f[0],f[1]) for f in frames]))

    def pose_preview(self,frame):
        return self._request(dict(op='pose_preview',frames=[(frame[0],frame[1])]))

    def _request(self,request):
        with self.lock:
            if not self.process or self.process.poll() is not None:raise RuntimeError('动作推理进程已退出')
            self.connection.send(request)
            if not self.connection.poll(20):raise TimeoutError('动作推理超过20秒，已停止接收新任务')
            return self.connection.recv()

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
        if self.connection:
            self.connection.close();self.connection=None


class LiveActionService:
    def __init__(self,settings,cameras,events,broadcast,simulation):
        self.settings,self.cameras,self.events,self.broadcast=settings,cameras,events,broadcast
        self.enabled=bool(settings.live_actions_enabled and not simulation)
        self.state='starting' if self.enabled else 'disabled'
        self.reason=None;self.worker=None;self.task=None;self.pending=set()
        self.camera_status={};self.runtime_info={}

    async def start(self):
        if not self.enabled:return
        self.worker=InferenceWorker(self.settings.live_actions_python,
            self.settings.resolve_path(self.settings.live_actions_manifest),self.settings.base_dir,self.settings.logs_dir)
        try:
            self.runtime_info=await asyncio.to_thread(self.worker.start)
            self.state='ready'
            self.task=asyncio.create_task(self.run(),name='live-actions')
        except Exception as exc:
            self.state='error';self.reason=str(exc)
            await asyncio.to_thread(self.worker.close)

    async def stop(self):
        if self.worker:await asyncio.to_thread(self.worker.close)
        if self.task:
            self.task.cancel()
            try:await self.task
            except asyncio.CancelledError:pass
        if self.pending:await asyncio.gather(*self.pending,return_exceptions=True)
        self.state='stopped'

    def status(self):
        now=time.time();cameras={}
        for camera,worker in list(self.cameras.workers.items()):
            value=dict(self.camera_status.get(camera,dict(state='warming',actions=[],reason='等待连续画面')))
            if not worker.online:value.update(state='offline',reason='摄像头离线',actions=[])
            elif now-(value.get('updated_at') or now)>15:value.update(state='stale',reason='识别结果已过期',actions=[])
            cameras[camera]=value
        return dict(enabled=self.enabled,state=self.state,reason=self.reason,cameras=cameras,
            runtime=self.runtime_info,interval_seconds=2)

    async def create_event(self,action,camera,frames):
        try:
            diagnostics={key:action[key] for key in ('model','threshold','people_evidence',
                'predicted_action','predicted_action_score','usable_tracks','rising_tracks_excluded') if key in action}
            prediction=VideoPrediction(action['event_type'],action['score'],action['start'],action['end'],
                action['model_version'],action.get('guard_version'),diagnostics)
            await self.events.create(prediction,camera,frames)
        except Exception:
            import logging
            logging.getLogger(__name__).exception('动作候选建档失败 %s',camera)

    async def run(self):
        last={};last_preview={}
        while True:
            try:
                for camera,worker in list(self.cameras.workers.items()):
                    if not worker.online:continue
                    frames=worker.ring.snapshot()
                    if not frames or time.time()-frames[-1][0]>2:continue
                    key=(camera,id(worker))
                    if frames[-1][0]-last.get(key,-float('inf'))<2:continue
                    last={k:v for k,v in last.items() if k[0]!=camera or k==key}
                    last[key]=frames[-1][0]
                    result=await asyncio.to_thread(self.worker.predict,f'{camera}:{id(worker)}',frames)
                    if self.cameras.get(camera) is not worker or not worker.online:continue
                    if time.time()-frames[-1][0]>15:
                        result=dict(state='stale',reason='处理耗时过长，已丢弃过期结果',actions=[])
                    result['updated_at']=time.time();self.camera_status[camera]=result
                    self.broadcast('action_status',dict(camera_id=camera,**result))
                    for action in result.get('actions',[]):
                        if action['state']=='candidate' and len(self.pending)<4:
                            task=asyncio.create_task(self.create_event(action,camera,frames))
                            self.pending.add(task);task.add_done_callback(self.pending.discard)
                # Serialize previews with action inference; no growing GPU queue.
                for camera,worker in list(self.cameras.workers.items()):
                    if not worker.online:continue
                    frame=next((f for f in reversed(worker.ring.snapshot()) if len(f)>2 and 'annotated' in f[2]),None)
                    if frame is None or time.time()-frame[0]>1:continue
                    key=(camera,id(worker))
                    if frame[0]-last_preview.get(key,-float('inf'))<.12:continue
                    last_preview={k:v for k,v in last_preview.items() if k[0]!=camera or k==key}
                    last_preview[key]=frame[0]
                    result=await asyncio.to_thread(self.worker.pose_preview,frame)
                    if self.cameras.get(camera) is worker and 'keypoints' in result:
                        await asyncio.to_thread(worker.set_pose_preview,frame,result)
                await asyncio.sleep(.02)
            except asyncio.CancelledError:raise
            except Exception as exc:
                self.state='error';self.reason=str(exc)
                await asyncio.to_thread(self.worker.close)
                return
