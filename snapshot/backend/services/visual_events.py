"""Persisted candidate lifecycle. Only a validated confirmation can request an alarm."""
from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path

from ..ai.queue import ReviewTask
from ..ai.qwen_reviewer import QwenResult
from ..ai.visual_review import validate_visual
from ..ai.audio_review import validate_audio
from ..audio.capture import save_audio
from ..schemas.events import RiskEventOut, EventFrameOut
from ..schemas.enums import EventSource, RiskLevel
from ..vision.video_events import sample_window
from ..vision.annotator import encode_jpeg

log = logging.getLogger(__name__)
LABELS = {"pending_review":"待复核", "confirmed":"已确认", "rejected":"已排除", "manual_review":"待人工核查"}

def review_state(result, event_type, audio=None):
    payload = {k: v for k, v in (result.payload or {}).items() if k != 'audio_assessment'} if audio is not None else result.payload
    valid, reason = validate_visual(payload, event_type)
    if valid and audio is not None:
        valid, reason = validate_audio(result.payload.get('audio_assessment'), audio)
    if not result.ok or not valid:
        return "manual_review", result.error or reason
    decision = result.payload["decision"]
    if audio is not None and (audio.get('status') != 'ready' or result.payload['audio_assessment']['status'] in {'conflicting', 'unavailable'}):
        return 'manual_review', '录音缺失、不完整或视听证据冲突，待人工核查'
    if decision == "uncertain":
        return "manual_review", "材料不足，待人工核查"
    return decision, None


class VisualEventService:
    def __init__(self, repo, events_dir, queue, broadcast, alarm, *, audio=None, cameras=None):
        self.repo, self.root, self.queue = repo, Path(events_dir), queue
        self.broadcast, self.alarm = broadcast, alarm
        self._lock = asyncio.Lock()
        self._last = {}
        self.audio, self.cameras = audio, cameras

    async def recover(self):
        # No replay of hardware side effects following a crash.
        await asyncio.to_thread(self.repo.db.execute_write,
            "UPDATE risk_events SET rule_basis=json_set(rule_basis,'$.visual.alarm_held',0) "
            "WHERE json_extract(rule_basis,'$.visual.state')='confirmed'")
        rows = await asyncio.to_thread(self.repo.db.query_all,
            "SELECT event_id FROM risk_events WHERE json_extract(rule_basis,'$.visual.state')='pending_review'")
        for row in rows:
            await self.on_result(row["event_id"], QwenResult("failed",error="服务重启，待人工核查"))

    async def create(self, prediction, camera_id, frames):
        key = camera_id, prediction.event_type
        if prediction.end - self._last.get(key, -float("inf")) < 30:
            return None
        selected = sample_window(frames,8)
        if len({f[0] for f in selected}) < 2:
            return None
        now = time.time()
        eid = "VIS-"+uuid.uuid4().hex
        title = "疑似人员倒地" if prediction.event_type == "person_fall" else "疑似打架"
        visual = {"state":"pending_review", "state_label":LABELS["pending_review"],
            "start":prediction.start,"end":prediction.end,"score":prediction.score,
            "model_version":prediction.model_version,"frame_times":[f[0] for f in selected],
            "guard_version":prediction.guard_version,
            "alarm_claimed":False,"alarm_held":False,"evidence_clip":None}
        if prediction.diagnostics:
            visual['diagnostics'] = prediction.diagnostics
            visual['threshold'] = prediction.diagnostics.get('threshold')
        event = RiskEventOut(event_id=eid,event_type=prediction.event_type,risk_level=RiskLevel.HIGH,
            event_type_label=title,
            source=EventSource.VISION,camera_id=camera_id,occurred_at=now,title=title,
            description=title+"，等待多模态确认",rule_basis={"visual":visual},created_at=now,updated_at=now)
        await asyncio.to_thread(self.repo.insert_event,event)
        self._last[key] = prediction.end
        self.broadcast("risk_event_created",event.model_dump(mode="json"))
        try:
            evidence = None
            if self.audio:
                await asyncio.sleep(max(0, prediction.end + self.audio.settings.audio_post_seconds - time.time()))
                worker = self.cameras.get(camera_id) if self.cameras else None
                combined = {f[0]: f for f in frames}
                if worker:
                    combined.update({f[0]: f for f in worker.ring.snapshot()})
                start = prediction.end - self.audio.settings.audio_pre_seconds
                end = prediction.end + self.audio.settings.audio_post_seconds
                frames = sorted((f for ts, f in combined.items() if start <= ts <= end), key=lambda f:f[0])
                selected = sample_window(frames, 8, seconds=end-start)
                if len(selected) != 8 or selected[-1][0] <= selected[0][0]:
                    raise ValueError('缺少有效时间窗内的图像')
                visual['frame_times'] = [f[0] for f in selected]
                visual['requested_window'] = {'start': start, 'end': end}
                evidence = await asyncio.to_thread(self.audio.evidence, selected[0][0], selected[-1][0], camera_id)
            images, annotated = await asyncio.to_thread(self._save,eid,camera_id,frames,selected)
            current = await asyncio.to_thread(self.repo.get_event, eid)
            basis = current.rule_basis
            basis['visual'].update(frame_times=visual['frame_times'], requested_window=visual.get('requested_window'))
            if evidence:
                folder = self.root / Path(current.raw_image).parent
                basis['audio'] = await asyncio.to_thread(save_audio, evidence, folder, self.root)
            await asyncio.to_thread(self.repo.update_event, eid, rule_basis=basis)
            self.broadcast('risk_event_updated', {'event_id': eid, 'rule_basis': basis})
            context={"review_mode":"audiovisual_candidate" if self.audio else "visual_candidate", "event_type":prediction.event_type,
                "frame_times":visual["frame_times"],"local_score":prediction.score,
                "camera_id":camera_id, "annotation_available":[bool(x) for x in annotated]}
            if evidence:
                context['audio'] = basis['audio']
            if self.queue is None or not self.queue.submit(ReviewTask(eid,context,images,
                audio_wav=evidence.wav if evidence else None, annotated_images=annotated)):
                await self.on_result(eid,QwenResult("skipped",error="复核未入队"))
        except Exception as exc:
            await self.on_result(eid,QwenResult("failed",error=f"证据保存失败: {exc}"))
        return eid

    def _save(self,eid,camera,frames,selected):
        import cv2
        folder=self.root/time.strftime("%Y%m%d")/eid
        folder.mkdir(parents=True,exist_ok=True)
        images=[]; annotated=[]
        for i,(ts,im,*rest) in enumerate(selected):
            meta = rest[0] if rest else {}
            im = meta.get('evidence_raw', im)
            payload=encode_jpeg(im)
            if not payload:
                raise ValueError("JPEG编码失败")
            path=folder/f"frame_{i:02}.jpg";path.write_bytes(payload);images.append(payload)
            rel=path.relative_to(self.root).as_posix()
            ann = encode_jpeg(meta['annotated']) if meta.get('annotated') is not None else None
            annotated.append(ann)
            ann_rel = None
            if ann:
                ann_path = folder / f'frame_{i:02}_annotated.jpg'
                ann_path.write_bytes(ann)
                ann_rel = ann_path.relative_to(self.root).as_posix()
            self.repo.insert_frame(EventFrameOut(event_id=eid,camera_id=camera,role="sequence",
                raw_path=rel,annotated_path=ann_rel,captured_at=ts,offset_seconds=ts-selected[0][0]))
        sampled=sample_window(frames,16,seconds=max(.1,selected[-1][0]-selected[0][0]))
        h,w=sampled[0][1].shape[:2]
        clip=folder/"evidence.avi"
        fps=15/max(.1,sampled[-1][0]-sampled[0][0])
        writer=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*"MJPG"),fps,(w,h))
        if not writer.isOpened():
            raise ValueError("视频编码器不可用")
        try:
            for _,im,*rest in sampled:
                meta = rest[0] if rest else {}
                writer.write(cv2.resize(meta.get('evidence_raw', im),(w,h)))
        finally:writer.release()
        e=self.repo.get_event(eid)
        basis=e.rule_basis;basis["visual"]["evidence_clip"]=clip.relative_to(self.root).as_posix()
        self.repo.update_event(eid,rule_basis=basis,raw_image=rel,annotated_image=ann_rel)
        return images, annotated

    async def on_result(self,eid,result):
        async with self._lock:
            event=await asyncio.to_thread(self.repo.get_event,eid)
            if not event or "visual" not in event.rule_basis:return False
            visual=event.rule_basis["visual"]
            if visual["state"] != "pending_review":return True
            state, reason = review_state(result, str(getattr(event.event_type,"value",event.event_type)), event.rule_basis.get('audio'))
            visual.update(state=state,state_label=LABELS[state],review_status=result.status,
                review_payload=result.payload,review_error=reason,
                reviewed_at=time.time(),alarm_claimed=state=="confirmed",alarm_held=state=="confirmed")
            visual.update(review_model=result.model, latency_ms=result.latency_ms, estimated_cost_cny=result.estimated_cost())
            # Atomic compare-and-set also protects against duplicate callbacks/processes.
            changed=await asyncio.to_thread(self.repo.db.execute_write,
                "UPDATE risk_events SET rule_basis=?, description=?, updated_at=? WHERE event_id=? "
                "AND json_extract(rule_basis,'$.visual.state')='pending_review'",
                (json.dumps(event.rule_basis,ensure_ascii=False),event.title+"："+LABELS[state],time.time(),eid))
            if not changed:return True
            if state=="confirmed":
                try:
                    action=await self.alarm(event)
                except Exception as exc:
                    action={"ok":False,"status":"failed","error":str(exc)}
                await asyncio.to_thread(self.repo.update_event,eid,alert_result=action)
            updated=await asyncio.to_thread(self.repo.get_event,eid)
            self.broadcast("risk_event_updated",updated.model_dump(mode="json"))
            return True


async def analyze_cameras(cameras,detectors,service):
    """Single asynchronous worker bounds GPU concurrency and drops stale windows."""
    last={}
    while True:
        for camera,worker in list(cameras.workers.items()):
            if not worker.online:continue
            frames=worker.ring.snapshot()
            if not frames or time.time()-frames[-1][0]>2 or frames[-1][0]-last.get(camera,0)<1:continue
            if frames[-1][0]-frames[0][0]<3:continue
            last[camera]=frames[-1][0]
            for detector in detectors:
                try:
                    predictions=await asyncio.to_thread(detector.predict,frames)
                    for prediction in predictions:await service.create(prediction,camera,frames)
                except Exception:
                    log.exception("视觉事件分析失败 %s",camera)
        await asyncio.sleep(.1)
