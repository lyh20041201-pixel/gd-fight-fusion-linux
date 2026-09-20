"""Frozen comparison of public FGN weights and the current local R3D-18.

Prepare -> verify -> validation inference -> validation-only calibration -> test.
No cloud calls, training, deployment, or test-driven threshold selection.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.update(YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false', HF_HUB_OFFLINE='1')

import cv2
import numpy as np
import torch
from scripts.rwf2000_model import load_model, VideoInputs, score, file_sha, WEIGHTS
from scripts.skeleton_common import digest, isolation, offline
from scripts.skeleton_round2 import choose_threshold
from scripts.evaluate_qwen_blind_ab import metrics
from backend.vision.action_guards import fight_people_evidence, FIGHT_PEOPLE_POLICY, GUARD_VERSION
from backend.vision.live_actions import rgb_tensor

OUT = ROOT / 'results/rwf2000_comparison/20260918_v1'
MANIFEST = ROOT / 'datasets/video_events/skeleton_rebuild/round2/manifests/vfd.json'
CACHE = ROOT / 'datasets/video_events/skeleton_rebuild/vfd'
CONFIG = ROOT / 'config/live_actions.json'
SELECTION = ROOT / 'results/video_events/skeleton_comparison_round2/vfd/rgb/seed_42/selection.json'
VALIDATION_BASELINE = ROOT / 'results/live_actions/fight_people_v3/validation.json'
TEST_BASELINE = ROOT / 'results/video_events/skeleton_comparison_round2/evaluations/vfd/retained_test/samples'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def prepare():
    if (OUT / 'protocol.json').exists():
        raise ValueError('Protocol already frozen; use run or report')
    config, source = read(CONFIG), read(MANIFEST)
    isolation(source['rows'])
    rows = [r for r in source['rows'] if r['split'] in ('validation', 'test')]
    assert Counter(r['split'] for r in rows) == {'validation': 207, 'test': 205}
    local = config['fight']['models'][0]
    assert file_sha(local['path']) == local['sha256']
    prior = {r['sample_id']: r for r in read(VALIDATION_BASELINE)['rows']}
    frozen = []
    for i, row in enumerate(rows):
        cache_path = CACHE / (row['sample_id'] + '.pt')
        cached = torch.load(cache_path, map_location='cpu', weights_only=True)
        assert cached['source_sha256'] == row['sha256']
        assert cached['pose_sha256'] == config['pose']['sha256']
        assert cached['status'] == 'complete'
        windows = []
        for c in cached['clips']:
            evidence = fight_people_evidence(c)
            windows.append(dict(start=c['start'], end=c['end'], eligible=evidence['eligible']))
        if row['split'] == 'validation':
            old = prior[row['sample_id']]
            old_scores = [c['score'] for c in old['clips']]
            old_raw = old['raw']
            baseline_sha = file_sha(VALIDATION_BASELINE)
        else:
            baseline_path = TEST_BASELINE / (row['sample_id'] + '.json')
            old_record = read(baseline_path)
            assert old_record['source_sha256'] == row['sha256'] and old_record['label'] == row['label']
            old = old_record['comparisons']['round2']['rgb_42']
            assert old['threshold'] == config['fight']['threshold']
            old_scores = [c['score'] for c in old['windows']]
            old_raw = old['score']
            baseline_sha = file_sha(baseline_path)
        assert len(windows) == len(old_scores)
        frozen.append({**row, 'windows': windows, 'cache_path': str(cache_path),
                       'cache_sha256': file_sha(cache_path), 'baseline_scores': old_scores,
                       'baseline_raw': old_raw, 'baseline_file_sha256': baseline_sha})
    frozen.sort(key=lambda r: (r['split'], digest(['rwf-public-model-v1', r['sample_id']])))
    write(OUT / 'manifest.json', {'rows': frozen})
    code_files = [Path(__file__), ROOT / 'scripts/rwf2000_model.py',
                  ROOT / 'scripts/skeleton_round2.py', ROOT / 'backend/vision/action_guards.py',
                  ROOT / 'backend/vision/live_actions.py']
    protocol = dict(
        created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='frozen_before_model_predictions',
        dataset='Existing VFD split; 207 validation + 205 retained test',
        counts={s: dict(Counter(str(r['label']) for r in frozen if r['split'] == s))
                for s in ('validation', 'test')},
        source_manifest_sha256=file_sha(MANIFEST), manifest_sha256=file_sha(OUT / 'manifest.json'),
        current_config_sha256=file_sha(CONFIG), current_checkpoint=local,
        current_threshold=config['fight']['threshold'], rwf_provenance=read(ROOT / 'models/rwf2000/provenance.json'),
        rwf_architecture='Author HDF5 graph, Keras 3 torch backend, all 50 learned tensors loaded exactly',
        current_inference='Current R3D-18 with live rgb_tensor and layer3 fp16 boundary, CUDA autocast fp16',
        rwf_inference='float32, TF32 disabled, no augmentation, no training',
        intervals='Exactly the existing 4-second windows, stride 2 seconds, short videos retained',
        rwf_preprocess='Native FPS, RGB INTER_AREA stretch224; author drop-last frame, adjacent Farneback, '
                       'mean subtraction and per-component minmax0-255, uint8 boundary, ceil-stride64/end padding, '
                       'separate RGB/flow standardization; final flow in each window zero',
        rwf_class_order=['Fight', 'NonFight'], rwf_fight_index=0, rwf_default_threshold=.5,
        rwf_threshold_policy='Validation only: exact existing choose_threshold, FPR <= .05; maximize recall, '
                             'then lower FPR, macro F1, higher threshold. Separately for raw and same-person-gate.',
        aggregation='Maximum window score per video; filtered variant only eligible windows; no eligible -> unknown',
        errors='Any failed required window makes the video unknown, never dropped; positive unknown counts as missed',
        metrics='P=TP/(TP+FP); R=TP/all positive; FPR=FP/all normal; F1=2TP/(2TP+FP+FN including unknown positives)',
        gate={'version': GUARD_VERSION, 'policy': FIGHT_PEOPLE_POLICY},
        timing='Synchronized GPU model call per window, with transfer; separate decode/optical-flow/preprocess timings. '
               'Baseline decoded RGB/YOLO cache reading is not end-to-end live latency.',
        training=False, cloud_calls=False, deployment=False,
        limitations=[
            'Previously used validation and retained test; not a newly collected blind campus evaluation.',
            'Current R3D-18 was trained/adapted on VFD; FGN is an external public checkpoint without local fine-tuning.',
            'RWF/VFD source-video overlap cannot be excluded without the original RWF source inventory.',
            'Public checkpoint has no embedded training manifest/epoch; provenance established to author repository only.',
            'Inherited VFD labels have not been independently reannotated for this comparison.',
            'Matched four-second windows differ from the approximately five-second source clips of RWF.',
            '64x224 RGB+flow versus 32x112 RGB compares complete model pipelines, not an isolated optical-flow ablation.',
            'Offline maximum over all windows uses future context; per-video FPR is not false alarms per hour.',
        ], code_sha256={str(p): file_sha(p) for p in code_files})
    write(OUT / 'protocol.json', protocol)
    print(json.dumps({'prepared': len(frozen), 'windows': sum(len(r['windows']) for r in frozen),
                      'counts': protocol['counts']}, ensure_ascii=False), flush=True)


def check_seal():
    protocol = read(OUT / 'protocol.json')
    assert file_sha(CONFIG) == protocol['current_config_sha256']
    assert file_sha(MANIFEST) == protocol['source_manifest_sha256']
    assert file_sha(OUT / 'manifest.json') == protocol['manifest_sha256']
    for path, expected in protocol['code_sha256'].items():
        assert file_sha(path) == expected, path
    assert file_sha(WEIGHTS) == protocol['rwf_provenance']['model_sha256']
    assert file_sha(protocol['current_checkpoint']['path']) == protocol['current_checkpoint']['sha256']
    return protocol


def init_runtime():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def load_baseline(protocol):
    from torchvision.models.video import r3d_18
    model = r3d_18(weights=None)
    model.fc = torch.nn.Linear(512, 2)
    saved = torch.load(protocol['current_checkpoint']['path'], map_location='cpu', weights_only=True)
    model.load_state_dict(saved['state_dict'])
    return model.cuda().eval()


def baseline_score(model, rgb):
    with torch.inference_mode(), torch.autocast('cuda'):
        value = rgb_tensor(rgb).unsqueeze(0).cuda()
        z = model.layer3(model.layer2(model.layer1(model.stem(value))))
        z = model.avgpool(model.layer4(z.half().float())).flatten(1)
        logits = model.fc(z)
        return float((logits[0, 1] - logits[0, 0]).sigmoid())


def prepare_video(row):
    try:
        return VideoInputs(row, [(c['start'], c['end']) for c in row['windows']]), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def aggregate(clips, field, gated=False):
    required = [c for c in clips if not gated or c['eligible']]
    if any(c[field] is None for c in required):
        return None
    values = [c[field] for c in required]
    return max(values) if values else None


def run(split, limit=None):
    protocol = check_seal()
    verify = read(OUT / 'verification.json')
    assert verify['passed'] and verify['model_sha256'] == file_sha(WEIGHTS)
    if split == 'test':
        calibration = read(OUT / 'calibration.json')
        assert calibration['protocol_sha256'] == file_sha(OUT / 'protocol.json')
    init_runtime()
    fgn = load_model()
    baseline = load_baseline(protocol)
    offline()
    # Warm-up is synthetic, label-independent and excluded from measured latency.
    score(fgn, np.zeros((64, 224, 224, 5), np.float32))
    baseline_score(baseline, np.zeros((32, 112, 112, 3), np.uint8))
    rows = [r for r in read(OUT / 'manifest.json')['rows'] if r['split'] == split]
    pending = [r for r in rows if not (OUT / split / 'samples' / (r['sample_id'] + '.json')).exists()]
    if limit:
        pending = pending[:limit]
    start = time.perf_counter()
    if not pending:
        print('No pending samples', split, flush=True)
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(prepare_video, pending[0])
        for index, row in enumerate(pending):
            inputs, input_error = future.result()
            if index + 1 < len(pending):
                future = pool.submit(prepare_video, pending[index + 1])
            if file_sha(row['cache_path']) != row['cache_sha256']:
                raise ValueError('Cached source changed')
            cache = torch.load(row['cache_path'], map_location='cpu', weights_only=True)
            records = []
            for j, (spec, cached) in enumerate(zip(row['windows'], cache['clips'], strict=True)):
                torch.cuda.synchronize()
                t = time.perf_counter()
                rgb_score = baseline_score(baseline, cached['rgb'])
                torch.cuda.synchronize()
                rgb_seconds = time.perf_counter() - t
                rwf_score, probabilities, error, audit = None, None, input_error, None
                rwf_seconds, transform_seconds = None, None
                if inputs is not None:
                    try:
                        t = time.perf_counter()
                        data, audit = inputs.window(j)
                        transform_seconds = time.perf_counter() - t
                        torch.cuda.synchronize()
                        t = time.perf_counter()
                        rwf_score, probabilities = score(fgn, data)
                        torch.cuda.synchronize()
                        rwf_seconds = time.perf_counter() - t
                        del data
                    except (ValueError, RuntimeError) as exc:
                        error = f'{type(exc).__name__}: {exc}'
                records.append({**spec, 'r3d_score': rgb_score, 'rwf_score': rwf_score,
                                'rwf_probabilities': probabilities, 'rwf_error': error,
                                'rwf_input': audit, 'r3d_inference_seconds': rgb_seconds,
                                'rwf_inference_seconds': rwf_seconds,
                                'rwf_transform_seconds': transform_seconds,
                                'r3d_prior_score': row['baseline_scores'][j]})
            result = dict(sample_id=row['sample_id'], label=row['label'], group=row['group'],
                          path=row['path'], source_sha256=row['sha256'], split=split,
                          protocol_sha256=file_sha(OUT / 'protocol.json'), windows=records,
                          r3d_raw=aggregate(records, 'r3d_score'), r3d_gated=aggregate(records, 'r3d_score', True),
                          rwf_raw=aggregate(records, 'rwf_score'), rwf_gated=aggregate(records, 'rwf_score', True),
                          rwf_decode_seconds=inputs.decode_seconds if inputs else None,
                          rwf_flow_seconds=inputs.flow_seconds if inputs else None,
                          rwf_preparation_seconds=inputs.preparation_seconds if inputs else None,
                          input_error=input_error,
                          baseline_max_abs_difference=max(abs(c['r3d_score'] - c['r3d_prior_score']) for c in records))
            write(OUT / split / 'samples' / (row['sample_id'] + '.json'), result)
            del cache, inputs
            completed = len(list((OUT / split / 'samples').glob('*.json')))
            progress = dict(split=split, completed=completed, total=len(rows),
                            elapsed_seconds=time.perf_counter() - start,
                            latest_sample=row['sample_id'], latest_windows=len(records))
            write(OUT / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    check_seal()


def completed_rows(split):
    manifest = [r for r in read(OUT / 'manifest.json')['rows'] if r['split'] == split]
    rows = []
    for source in manifest:
        record = read(OUT / split / 'samples' / (source['sample_id'] + '.json'))
        assert record['protocol_sha256'] == file_sha(OUT / 'protocol.json')
        assert record['source_sha256'] == source['sha256'] and record['label'] == source['label']
        rows.append(record)
    return rows


def predictions(rows, field, threshold):
    return [-1 if r[field] is None else int(r[field] >= threshold) for r in rows]


def calibrate():
    check_seal()
    if (OUT / 'calibration.json').exists():
        raise ValueError('Calibration already sealed')
    assert not (OUT / 'test/samples').exists(), 'Test already accessed before calibration'
    rows = completed_rows('validation')
    result = dict(protocol_sha256=file_sha(OUT / 'protocol.json'), validation_samples=len(rows),
                  sample_files_sha256={r['sample_id']: file_sha(OUT / 'validation/samples' / (r['sample_id'] + '.json')) for r in rows},
                  created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    for field in ('rwf_raw', 'rwf_gated'):
        threshold, selected = choose_threshold([r['label'] for r in rows], [r[field] for r in rows])
        result[field] = dict(threshold=threshold, selection_metrics=selected,
                            default_metrics=metrics([r['label'] for r in rows], predictions(rows, field, .5)))
    write(OUT / 'calibration.json', result)
    print(json.dumps({k: result[k] for k in ('rwf_raw', 'rwf_gated')}, ensure_ascii=False), flush=True)


def report():
    protocol = check_seal()
    calibration = read(OUT / 'calibration.json')
    fields = {
        'R3D-18 单独': ('r3d_raw', protocol['current_threshold']),
        '现有方案：R3D-18＋人员筛选': ('r3d_gated', protocol['current_threshold']),
        '公开FGN 默认阈值0.5': ('rwf_raw', .5),
        '公开FGN 验证集校准阈值': ('rwf_raw', calibration['rwf_raw']['threshold']),
        '公开FGN＋同一人员筛选，校准阈值': ('rwf_gated', calibration['rwf_gated']['threshold']),
    }
    output = dict(protocol=protocol, calibration=calibration, splits={})
    lines = ['# RWF仓库公开预训练模型与现有R3D-18对比', '',
             '已下载作者公开FGN权重并完成离线对比；没有重新训练、调用Qwen或替换正在使用的模型。', '',
             '先在207段既有验证集上按既定规则选阈值，再固定阈值评估205段保留测试集。模型输入为相同时间窗口，保留各自的分辨率、抽帧和预处理。', '']
    flat = []
    for split in ('validation', 'test'):
        rows = completed_rows(split)
        summaries = {name: metrics([r['label'] for r in rows], predictions(rows, field, threshold))
                     for name, (field, threshold) in fields.items()}
        timing = {}
        for label, field in [('r3d', 'r3d_inference_seconds'), ('rwf', 'rwf_inference_seconds')]:
            values = [c[field] for r in rows for c in r['windows'] if c[field] is not None]
            timing[label] = dict(windows=len(values), median_seconds=float(np.median(values)),
                                 p95_seconds=float(np.percentile(values, 95)), total_seconds=sum(values))
        timing['rwf_preparation_total_seconds'] = sum(r['rwf_preparation_seconds'] or 0 for r in rows)
        timing['rwf_flow_total_seconds'] = sum(r['rwf_flow_seconds'] or 0 for r in rows)
        timing['rwf_transform_total_seconds'] = sum(c['rwf_transform_seconds'] or 0 for r in rows for c in r['windows'])
        differences = [r['baseline_max_abs_difference'] for r in rows]
        output['splits'][split] = dict(samples=len(rows), metrics=summaries, timing=timing,
            baseline_max_abs_difference=max(differences), rows=rows)
        lines += ['## ' + ('验证集（207段，校准使用）' if split == 'validation' else '保留测试集（205段）'), '',
                  '| 方案 | TP | FP | FN | P 精确率 | R 召回率 | FPR 正常误报率 | F1 | 不确定 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        percent = lambda v: '—' if v is None else f'{v*100:.2f}%'
        for name, met in summaries.items():
            lines.append(f"| {name} | {met['tp']} | {met['fp']} | {met['fn']} | {percent(met['precision'])} | {percent(met['recall'])} | {percent(met['fpr'])} | {percent(met['f1'])} | {met['unknown']} |")
        lines += ['', f"当前R3D阈值：{protocol['current_threshold']:.10g}；FGN原始分数校准阈值：{calibration['rwf_raw']['threshold']:.10g}；FGN人员筛选后校准阈值：{calibration['rwf_gated']['threshold']:.10g}。", '',
                  f"每个窗口的模型调用耗时中位数：R3D {timing['r3d']['median_seconds']*1000:.1f} ms，FGN {timing['rwf']['median_seconds']*1000:.1f} ms；均同步GPU，FGN另有光流计算和预处理开销。", '',
                  f"本批FGN光流计算累计 {timing['rwf_flow_total_seconds']:.1f} 秒，解码/校验/光流准备累计 {timing['rwf_preparation_total_seconds']:.1f} 秒。准备与GPU推理有流水重叠，累计值不是总墙钟时长。", '',
                  f"本次重跑R3D与既有缓存分数最大绝对差：{max(differences):.8g}。", '']
        for r in rows:
            flat.append(dict(split=split, sample_id=r['sample_id'], label=r['label'], path=r['path'],
                             **{k: r[k] for k in ('r3d_raw', 'r3d_gated', 'rwf_raw', 'rwf_gated')}))
    lines += ['## 指标口径与使用范围', '',
              '- P：报告打架的视频中，来源标签为打架的比例。',
              '- R：所有来源标签为打架的视频中，被检出的比例。',
              '- FPR：所有来源标签为正常的视频中，被误报为打架的比例，不是1−P。',
              '- 未能判断保留在分母中；正例未能判断计漏检，正常未能判断不当作正确识别。',
              '- 这些是视频级结果，不能解释为每小时误报次数或摄像头实时报警延迟。',
              '- R3D使用已训练于VFD的现有权重；FGN仅做验证集阈值校准，未在VFD上微调。',
              '- 现有验证集和保留测试集此前已被使用，不是新采集的独立校园盲测；两种数据源可能重叠，目前无法排除。',
              '- 源标签沿用VFD，未因模型输出而更改。FGN的原始训练清单/训练轮次无法从公开文件独立核实。',
              '- 光流与RGB模型的输入尺寸、帧数和结构不同，不能把差异全部归因于光流。', '',
              '下载模型：`E:/GD/models/rwf2000/keras_model.h5`；来源、文件SHA256见同目录provenance.json。',
              '完整逐样本和逐窗口记录见evaluation.json，简表见predictions.csv。']
    write(OUT / 'evaluation.json', output)
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with (OUT / 'predictions.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({s: output['splits'][s]['metrics'] for s in output['splits']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'calibrate', 'report'])
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    {'prepare': prepare, 'run': lambda: run(args.split, args.limit), 'calibrate': calibrate, 'report': report}[args.command]()
