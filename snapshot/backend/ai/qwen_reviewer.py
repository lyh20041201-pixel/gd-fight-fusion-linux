"""Qwen3.5-Omni on DashScope: image pairs, raw WAV and context in one request."""
from __future__ import annotations
import asyncio
import base64
import copy
import io
import json
import time
import wave
from .prompts import SYSTEM_PROMPT, RISK_REVIEW_SCHEMA, validate_review
from .visual_review import VISUAL_SCHEMA, VISUAL_PROMPT, validate_visual
from .audio_review import AUDIO_SCHEMA, AUDIO_PROMPT, validate_audio

MODELS = {'qwen3.5-omni-flash', 'qwen3.5-omni-plus'}
# CNY / million tokens, Beijing list prices checked 2026-09-14.
PRICES = {'qwen3.5-omni-flash': (2.2, 18., 13.3), 'qwen3.5-omni-plus': (7., 53., 40.)}

class QwenResult:
    def __init__(self, status, payload=None, error=None, *, latency_ms=0., prompt_tokens=None,
                 completion_tokens=None, raw=None, model=None, audio_tokens=None):
        self.status, self.payload, self.error = status, payload or {}, error
        self.latency_ms, self.prompt_tokens, self.completion_tokens = latency_ms, prompt_tokens, completion_tokens
        self.raw, self.model, self.audio_tokens = raw, model, audio_tokens

    @property
    def ok(self):
        return self.status == 'ok'

    def estimated_cost(self):
        if self.model not in PRICES or any(x is None for x in (self.prompt_tokens, self.completion_tokens, self.audio_tokens)):
            return None
        image, audio, output = PRICES[self.model]
        return round((max(0, self.prompt_tokens-self.audio_tokens)*image + self.audio_tokens*audio
                      + self.completion_tokens*output)/1e6, 6)

class QwenReviewer:
    def __init__(self, settings, runtime):
        self.settings, self.runtime = settings, runtime
        self._client = None
        self._last_call_at = None
        self.calls = self.failures = 0
        self.total_cost = 0.
        self.last_error = self.last_success_at = None

    @property
    def configured(self): return self.settings.qwen_configured
    @property
    def enabled(self): return self.runtime.qwen_enabled
    @property
    def available(self): return self.configured and self.enabled and self.last_error is None

    def status(self):
        return {'provider': 'dashscope', 'configured': self.configured, 'enabled': self.enabled,
                'available': self.available, 'model': self.runtime.qwen_model, 'calls': self.calls,
                'failures': self.failures, 'estimated_cost': round(self.total_cost, 6), 'currency': 'CNY',
                'cost_basis': '北京地域参考价；以百炼账单为准', 'last_error': self.last_error,
                'last_success_at': self.last_success_at, 'cooldown_seconds': self.settings.qwen_cooldown_seconds}

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            # The SDK is a protocol client, with no OpenAI endpoint/key fallback.
            self._client = AsyncOpenAI(api_key=self.settings.dashscope_api_key,
                base_url=self.settings.dashscope_base_url, timeout=self.settings.qwen_timeout_seconds, max_retries=0)
        return self._client

    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None

    async def review(self, context, images=None, audio_wav=None, annotated_images=None):
        model = self.runtime.qwen_model
        if not self.enabled or not self.configured:
            return QwenResult('skipped', error='Qwen 已停用' if not self.enabled else '未配置 DASHSCOPE_API_KEY', model=model)
        if model not in MODELS:
            return QwenResult('failed', error='仅支持 Qwen3.5-Omni Flash / Plus', model=model)
        if self._last_call_at is not None and time.monotonic()-self._last_call_at < self.settings.qwen_cooldown_seconds:
            return QwenResult('skipped', error='复核调用冷却中', model=model)
        visual = context.get('review_mode') in {'visual_candidate', 'audiovisual_candidate'}
        av = context.get('review_mode') == 'audiovisual_candidate' or 'audio' in context
        images, annotated_images = images or [], annotated_images or []
        if visual and (len(images) != 8 or len(context.get('frame_times', [])) != 8):
            return QwenResult('failed', error='视觉复核需要8帧及对应时间戳', model=model)
        if annotated_images and len(annotated_images) != len(images):
            return QwenResult('failed', error='原图与标注图数量不匹配', model=model)
        audio_meta = context.get('audio', {})
        if bool(audio_wav) != (audio_meta.get('status') in {'ready', 'partial'}):
            return QwenResult('failed', error='音频内容与采集状态不一致', model=model)
        if audio_wav:
            try:
                with wave.open(io.BytesIO(audio_wav), 'rb') as wav:
                    duration = wav.getnframes()/wav.getframerate()
                    if (wav.getnchannels() != 1 or wav.getsampwidth() != 2 or not 0 < duration <= 30
                            or abs(duration-(audio_meta['end']-audio_meta['start'])) > .02):
                        raise ValueError('invalid WAV')
            except Exception:
                return QwenResult('failed', error='WAV格式或时间戳无效', model=model)
        schema = copy.deepcopy(VISUAL_SCHEMA if visual else RISK_REVIEW_SCHEMA)
        if av:
            schema['required'].append('audio_assessment')
            schema['properties']['audio_assessment'] = AUDIO_SCHEMA
        prompt = (VISUAL_PROMPT if visual else SYSTEM_PROMPT) + AUDIO_PROMPT + '\nJSON Schema:\n' + json.dumps(schema, ensure_ascii=False)
        content = [{'type': 'text', 'text': json.dumps(context, ensure_ascii=False)}]
        for i, picture in enumerate(images[:8 if visual else self.settings.qwen_max_frames]):
            content.append({'type': 'text', 'text': f'frame_index={i} 原图；时间见 frame_times / image_manifest'})
            content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(picture).decode()}})
            if annotated_images and annotated_images[i]:
                content.append({'type': 'text', 'text': f'frame_index={i} 同帧本地标注，仅辅助定位'})
                content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(annotated_images[i]).decode()}})
        if audio_wav:
            content.append({'type': 'input_audio', 'input_audio': {
                'data': 'data:;base64,' + base64.b64encode(audio_wav).decode(), 'format': 'wav'}})
        started = time.perf_counter()
        for attempt in range(self.settings.qwen_max_retries + 1):
            try:
                client = self._ensure_client()
                self._last_call_at = time.monotonic()
                self.calls += 1
                async with asyncio.timeout(self.settings.qwen_timeout_seconds):
                    stream = await client.chat.completions.create(model=model,
                        messages=[{'role': 'system', 'content': prompt}, {'role': 'user', 'content': content}],
                        modalities=['text'], stream=True, stream_options={'include_usage': True},
                        max_tokens=self.settings.qwen_max_output_tokens)
                    parts, usage, finish, length = [], None, None, 0
                    try:
                        async for chunk in stream:
                            if chunk.usage: usage = chunk.usage
                            if chunk.choices:
                                choice = chunk.choices[0]
                                if choice.delta.content:
                                    parts.append(choice.delta.content)
                                    length += len(choice.delta.content)
                                finish = choice.finish_reason or finish
                                if length > 24000: raise ValueError('response too long')
                    finally:
                        await stream.close()
                result = self._parse(''.join(parts), context, visual, av)
                result.model, result.latency_ms = model, (time.perf_counter()-started)*1000
                result.prompt_tokens = getattr(usage, 'prompt_tokens', None)
                result.completion_tokens = getattr(usage, 'completion_tokens', None)
                details = getattr(usage, 'prompt_tokens_details', None)
                result.audio_tokens = (details.get('audio_tokens') if isinstance(details, dict) else getattr(details, 'audio_tokens', None)) if audio_wav else 0
                if finish != 'stop': result.status, result.error = 'failed', '模型输出未正常结束'
                cost = result.estimated_cost()
                if cost is not None: self.total_cost += cost
                self.last_error = result.error
                if result.ok: self.last_success_at = time.time()
                else: self.failures += 1
                return result
            except (TimeoutError, asyncio.TimeoutError):
                status, reason, retry = 'timeout', 'Qwen 复核超时', True
            except Exception as exc:
                code = getattr(exc, 'status_code', None)
                # SDK exception strings may contain request bodies, URLs or credentials.
                status = 'timeout' if type(exc).__name__ == 'APITimeoutError' else 'failed'
                reason = f'Qwen 调用失败：HTTP {code}' if code else f'Qwen 调用失败：{type(exc).__name__}'
                retry = code in {408, 429, 500, 502, 503, 504} or type(exc).__name__ in {'APIConnectionError', 'APITimeoutError'}
            if not retry or attempt == self.settings.qwen_max_retries: break
            await asyncio.sleep(min(2*(attempt+1), 4))
        self.failures += 1
        self.last_error = reason
        return QwenResult(status, error=reason, model=model, latency_ms=(time.perf_counter()-started)*1000)

    def _parse(self, raw, context, visual, av):
        try: payload = json.loads(raw)
        except (ValueError, TypeError): return QwenResult('failed', error='返回内容不是合法JSON', raw=raw[:24000])
        if not isinstance(payload, dict): return QwenResult('failed', error='复核结果必须为JSON对象')
        base = {k: v for k, v in payload.items() if k != 'audio_assessment'} if av else payload
        valid, reason = validate_visual(base, context.get('event_type')) if visual else validate_review(base)
        if valid and av: valid, reason = validate_audio(payload.get('audio_assessment'), context.get('audio'))
        if valid and av and (context.get('audio', {}).get('status') != 'ready' or payload['audio_assessment']['status'] in {'conflicting', 'unavailable'}):
            if visual: payload['decision'] = 'uncertain'
            else: payload['needs_human_review'] = True
        return QwenResult('ok' if valid else 'failed', payload=payload if valid else None, error=None if valid else reason, raw=raw[:24000])

    def _parse_response(self, response, latency_ms, visual_event=None):
        """For saved parser fixtures only; network uses Chat Completions."""
        result = self._parse(getattr(response, 'output_text', ''), {'event_type': visual_event}, bool(visual_event), False)
        result.latency_ms = latency_ms
        return result
