"""Independent-source stacking, sealed calibration, and untouched-test evaluation.

No test-set scores are generated until selection_seal.json has been written.
Existing production configuration is read as a baseline and never overwritten.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
from contextlib import nullcontext
import argparse
import json
import math
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import cv2
import torch
from torch.nn import functional as F
from backend.vision.fight_fusion import (make_rgb_model, FusionSkeletonModel, QualityFusion,
    fusion_features, FUSION_FEATURE_NAMES)
from scripts.fight_fusion_features import (OUT, FeatureStore, decode_window, atomic_json,
    atomic_torch, check_space, event_balanced_indices, window_label)
from scripts.skeleton_common import sha, digest, offline, seal
from scripts.skeleton_round2 import from_confusion, choose_threshold


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sigmoid(value):
    if value is None:
        return None
    return float(torch.tensor(value, dtype=torch.float32).sigmoid())


def count_metrics(labels, scores, threshold):
    if len(labels) != len(scores) or not math.isfinite(float(threshold)):
        raise ValueError('Metrics require matching labels/scores and a finite threshold')
    cm = np.zeros((2, 3), dtype=np.int64)
    for y, s in zip(labels, scores):
        if y not in (0, 1) or (s is not None and (not math.isfinite(float(s)) or not 0 <= s <= 1)):
            raise ValueError('Metrics require binary labels and finite probabilities or None')
        cm[int(y), 2 if s is None else int(s >= threshold)] += 1
    return from_confusion(cm)


def dominant_threshold(labels, scores, baseline_scores, baseline_threshold):
    """Use calibration only; do not quietly trade additional FPs for recall."""
    baseline = count_metrics(labels, baseline_scores, baseline_threshold)
    cm = np.asarray(baseline['confusion_matrix'])
    positives = sum(int(v) for v in labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError('Calibration must contain both source classes')
    candidates = sorted({float(s) for s in scores if s is not None} | {0., 1.0000001})
    measured = []
    for threshold in candidates:
        metrics = count_metrics(labels, scores, threshold)
        ncm = np.asarray(metrics['confusion_matrix'])
        key = (metrics['recall'][1], -metrics['normal_false_positive_rate'], metrics['macro_f1'], threshold)
        measured.append((threshold, metrics, key, int(ncm[1, 1]) > int(cm[1, 1]) and int(ncm[0, 1]) < int(cm[0, 1])))
    improved = [r for r in measured if r[3]]
    feasible = [r for r in measured if r[1]['normal_false_positive_rate'] <= baseline['normal_false_positive_rate']]
    chosen = max(improved or feasible or measured, key=lambda r: r[2])
    return dict(threshold=chosen[0], metrics=chosen[1], baseline=baseline,
                simultaneously_improved=bool(chosen[3]),
                policy='more TP and fewer FP than fixed production baseline; otherwise best recall without increased FPR, explicitly not passed')


def branch_selection(seed, manifest_sha256=None, feature_signature=None):
    lock_path = OUT/'branches/skeleton_arm_selection.json'
    lock = read(lock_path)
    if lock.get('seed') != 42 or lock.get('arm') not in ('skeleton_random', 'skeleton_ntu'):
        raise ValueError('A seed-42 skeleton initialization decision must be locked first')
    for entry in lock.get('source_selections', {}).values():
        if isinstance(entry, dict) and 'path' in entry and 'sha256' in entry and sha(entry['path']) != entry['sha256']:
            raise ValueError('Locked seed-42 skeleton selection changed')
    specs = {}
    arms = ('skeleton_random', 'skeleton_ntu') if seed == 42 else (lock['arm'],)
    for branch in ('global', 'roi', *arms):
        folder = OUT/'branches'/branch/f'seed{seed}'
        if not folder.exists():
            alternate = OUT/'branches'/branch/f'seed_{seed}'
            if alternate.exists():
                folder = alternate
        record = read(folder/'selection.json')
        path = folder/'selected_best.pt'
        if record.get('status', 'complete') != 'complete' or sha(path) != record['sha256']:
            raise ValueError(f'Incomplete or changed branch: {folder}')
        config = read(folder/'config.json')
        if record.get('smoke') or config.get('smoke') or record.get('test_accessed'):
            raise ValueError('Smoke or test-selected checkpoints cannot enter formal fusion')
        if manifest_sha256 is not None and config.get('manifest_sha256') != manifest_sha256:
            raise ValueError('Branch trained from a different manifest')
        if feature_signature is not None and config.get('feature_signature') != feature_signature:
            raise ValueError('Branch trained with different visual inputs')
        if record.get('selected_stage') not in ('head', 'finetune'):
            raise ValueError('Unknown selected branch training stage')
        specs[branch] = dict(path=str(path), sha256=sha(path), selection=record)
    def key(branch):
        m = specs[branch]['selection']['validation']
        return (m['recall'][1], -m['normal_false_positive_rate'], m['macro_f1'])
    # Deterministic tie preference to random initialization; never inspect test results.
    arm = max(arms, key=key)
    if arm != lock['arm']:
        raise ValueError('Seed-42 validation choice disagrees with the locked skeleton arm')
    return {k: specs[k] for k in ('global', 'roi')} | {'skeleton': specs[arm]}, arm


class CachedScorer:
    def __init__(self, specs, store):
        self.store = store
        self.device = store.device
        self.models = {}
        for name, spec in specs.items():
            if sha(spec['path']) != spec['sha256']:
                raise ValueError('Model digest mismatch')
            m = FusionSkeletonModel() if name == 'skeleton' else make_rgb_model(False)
            m.load_state_dict(torch.load(spec['path'], map_location='cpu', weights_only=True)['state_dict'])
            self.models[name] = m.to(self.device).eval()
        self.old_config = read(ROOT/'config/live_actions.json')
        self.old_threshold = self.old_config['fight']['threshold']
        self.old_models = []
        for spec in self.old_config['fight']['models']:
            if sha(spec['path']) != spec['sha256']:
                raise ValueError('Production baseline digest mismatch')
            m = make_rgb_model(False)
            m.load_state_dict(torch.load(spec['path'], map_location='cpu', weights_only=True)['state_dict'])
            self.old_models.append(m.to(self.device).eval())

    @torch.inference_mode()
    def rgb(self, model, features):
        z = features.unsqueeze(0).float().to(self.device)
        pooled = model.avgpool(model.layer4(z)).flatten(1)
        logits = model.fc(pooled)
        return logits[0, 1] - logits[0, 0]

    @torch.inference_mode()
    def score(self, row, window, include_baseline=False):
        context = torch.autocast('cuda') if self.device.type == 'cuda' else nullcontext()
        with context:
            g = float(self.rgb(self.models['global'], window['global_layer3']))
            local = [float(self.rgb(self.models['roi'], z)) for z in window['roi_layer3']]
            item = {k: v.to(self.device) for k, v in window['skeleton'].items()}
            sk = self.models['skeleton'](item)
            s = float(sk.float()) if sk is not None else None
            old = [float(self.rgb(m, window['global_layer3']).sigmoid()) for m in self.old_models] if include_baseline else []
        logits = [g, max(local) if local else None, s]
        record = dict(start=window['start'], end=window['end'], label=window['label'],
            logits=logits, quality=window['quality'], available=[v is not None for v in logits])
        if include_baseline:
            score = float(np.median(old)) if self.old_config['fight']['method'] == 'median' else float(np.mean(old))
            # Original detector threshold, tracking and people guard, evaluated on
            # the exact same sampled observations. This is a replay baseline,
            # not an assertion of measured operating-camera performance.
            images, times, indices, shape = decode_window(row, (window['start'], window['end']))
            self.store._init_models()
            from backend.vision.live_actions import PoseWindows
            from backend.vision.action_guards import fight_people_evidence
            frames = list(zip(times, images))
            observed = list({float(ts): (ts, image) for ts, image in frames}.values())
            old_clip = PoseWindows(self.store.pose).clip(str(row['sample_id']), observed, frames)
            eligible = fight_people_evidence(old_clip)['eligible']
            record.update(legacy_raw=score, legacy_gated=score if eligible else None)
        return record


def score_rows(rows, scorer, dest, signature, include_baseline=False):
    records = []
    dest = Path(dest)
    for i, row in enumerate(rows):
        path = dest/(row['sample_id']+'.json')
        wanted = digest([signature, row, include_baseline])
        if path.exists():
            saved = read(path)
            if saved['signature'] != wanted:
                raise ValueError('Window-score provenance mismatch')
            record = saved['record']
        else:
            check_space()
            started = time.monotonic()
            record = dict(sample_id=row['sample_id'], dataset=row['dataset'], split=row['split'],
                group=row['group'], label=row['label'], label_kind=row.get('label_kind', 'video'),
                duration=row['duration'], positive_intervals=row.get('positive_intervals'), windows=[])
            for window in scorer.store.get(row, full=True):
                record['windows'].append(scorer.score(row, window, include_baseline))
            record['elapsed_seconds'] = time.monotonic()-started
            atomic_json(path, dict(signature=wanted, record=record))
        records.append(record)
        atomic_json(OUT/'fusion_progress.json', dict(phase='score_'+row['split'], completed=i+1, total=len(rows), sample_id=row['sample_id']))
        print('SCORED', row['split'], i+1, '/', len(rows), row['sample_id'], flush=True)
    return records


def feature_matrix(record, variant):
    features = []
    for w in record['windows']:
        values, available = list(w['logits']), list(w['available'])
        if variant == 'global_roi':
            values[2] = None
            available[2] = False
        features.append(fusion_features(values, w['quality'], available))
    if not features:
        raise ValueError('No decoded windows; cannot omit source from fit')
    return torch.stack(features)


def dataset_source_weights(records):
    """Equal dataset mass, then equal source mass inside each dataset."""
    if not records:
        raise ValueError('Independent fusion-fit sources are empty')
    counts = Counter(r['dataset'] for r in records)
    return torch.tensor([1 / (len(counts) * counts[r['dataset']]) for r in records], dtype=torch.float32)


def frame_choices(record, epoch):
    labels = np.asarray([w['label'] for w in record['windows']])
    if not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError('Every temporally annotated fit window needs a binary label')
    spans = [(w['start'], w['end']) for w in record['windows']]
    if 'positive_intervals' not in record:
        raise ValueError('Event-balanced fit requires original temporal event boundaries')
    annotated = [window_label(record, span) for span in spans]
    if not np.array_equal(labels, annotated):
        raise ValueError('Cached fit window labels differ from annotated event boundaries')
    chosen = event_balanced_indices(spans, record['positive_intervals'], record['sample_id'], epoch,
                                    min(16, int((labels == 1).sum())), min(16, int((labels == 0).sum())))
    return chosen, labels


@torch.no_grad()
def normalize_dataset_sources(model, xs, weights, records=None):
    # Each source's bounded sample has exactly its assigned source weight,
    # independent of video duration or the number of videos in another dataset.
    means, seconds = [], []
    for index, x in enumerate(xs):
        choices = (frame_choices(records[index], 0)[0]
                   if records is not None and records[index]['label_kind'] == 'frame'
                   else np.linspace(0, len(x)-1, min(16, len(x))).astype(int))
        sample = x[choices]
        means.append(sample.mean(0))
        seconds.append(sample.square().mean(0))
    mean = (torch.stack(means) * weights[:, None]).sum(0)
    variance = ((torch.stack(seconds) * weights[:, None]).sum(0) - mean.square()).clamp_min(0)
    std = variance.sqrt()
    model.feature_mean.copy_(mean)
    model.feature_std.copy_(torch.where(std >= 1e-3, std, torch.ones_like(std)))


def fit_logistic(records, variant, seed, folder, signature):
    path = folder/(variant+'.pt')
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if (saved['signature'] != signature or saved.get('variant') != variant or saved.get('seed') != seed
                or tuple(saved.get('feature_names', ())) != FUSION_FEATURE_NAMES):
            raise ValueError('Fitted fusion signature mismatch')
        model = QualityFusion()
        model.load_state_dict(saved['state_dict'])
        return model.eval()
    weights = dataset_source_weights(records)
    supervised = []
    for record in records:
        if record['label_kind'] == 'frame':
            chosen, labels = frame_choices(record, 0)
            supervised.append(labels[chosen].tolist())
        else:
            supervised.append([record['label']])
    if {int(v) for labels in supervised for v in labels} != {0, 1}:
        raise ValueError('Independent fusion-fit groups require both labels')
    torch.manual_seed(seed)
    xs = [feature_matrix(r, variant) for r in records]
    model = QualityFusion(missing_dropout=.2)
    normalize_dataset_sources(model, xs, weights, records)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    prior = sum(float(weight) * float(np.mean(labels)) for weight, labels in zip(weights, supervised))
    weight = torch.tensor((1-prior)/max(prior, 1e-6))
    history = []
    # Fixed budget fixed before calibration: no calibration/test hyperparameter search.
    for epoch in range(250):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for i in torch.randperm(len(records)).tolist():
            record, x = records[i], xs[i]
            if record['label_kind'] == 'frame':
                # Rotate a bounded sample, retaining both window labels when present.
                chosen, ys = frame_choices(record, epoch)
                pred = model(x[chosen])
                target = torch.tensor(ys[chosen], dtype=torch.float32)
            else:
                pred = model(x).max()
                target = torch.tensor(float(record['label']))
            loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=weight)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite fusion objective')
            (loss * weights[i]).backward()
            total += float(loss.detach()) * float(weights[i])
        regularizer = .5 * .001 * model.linear.weight.square().sum()
        regularizer.backward()
        total += float(regularizer.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        history.append(total)
        if (epoch+1)%25 == 0:
            atomic_json(OUT/'fusion_progress.json', dict(phase='fit_'+variant, epoch=epoch+1, epochs=250, loss=history[-1]))
            print('FUSION', variant, epoch+1, history[-1], flush=True)
    model.eval()
    atomic_torch(path, dict(state_dict=model.state_dict(), signature=signature, feature_names=list(FUSION_FEATURE_NAMES),
                           variant=variant, seed=seed, epochs=250, loss_history=history,
                           source_weighting='equal dataset mass; equal source mass within dataset',
                           weighted_sampled_positive_prior=prior,
                           regularization='0.5 * 0.001 * squared linear weights; bias excluded'))
    return model


@torch.inference_mode()
def apply_models(records, models):
    result = []
    for record in records:
        row = dict(record)
        row['windows'] = [dict(w) for w in record['windows']]
        for variant, model in models.items():
            scores = model(feature_matrix(record, variant)).float().sigmoid().tolist()
            for window, score in zip(row['windows'], scores):
                window[variant] = score
        for w in row['windows']:
            w['global'] = sigmoid(w['logits'][0])
        row['scores'] = {}
        for name in ('legacy_gated', 'legacy_raw', 'global', *models):
            valid = [w.get(name) for w in row['windows'] if w.get(name) is not None]
            row['scores'][name] = max(valid) if valid else None
        result.append(row)
    return result


def continuous_metrics(records, key, threshold):
    events, found, normal_seconds, false_episodes = 0, 0, 0., 0
    delays = []
    for r in records:
        if r['label_kind'] != 'frame':
            continue
        intervals = r['positive_intervals']
        normal_seconds += max(0, r['duration']-sum(b-a for a,b in intervals))
        events += len(intervals)
        alarms = [w for w in r['windows'] if w.get(key) is not None and w[key] >= threshold]
        used = set()
        for a,b in intervals:
            hits = [(j,w) for j,w in enumerate(alarms) if j not in used and w['start'] < b and w['end'] > a]
            if hits:
                j,w = min(hits, key=lambda p:p[1]['end'])
                used.add(j)
                found += 1
                delays.append(max(0., w['end']-a))
        last = None
        for w in alarms:
            if any(w['start'] < b and w['end'] > a for a,b in intervals):
                continue
            if last is None or w['start'] > last:
                false_episodes += 1
            last = max(last or 0., w['end'])
    return dict(annotated_events=events, detected_events=found,
        event_recall=found/events if events else None, normal_hours=normal_seconds/3600,
        false_alarm_episodes=false_episodes,
        false_alarms_per_normal_hour=false_episodes/(normal_seconds/3600) if normal_seconds else None,
        median_observation_delay_seconds=float(np.median(delays)) if delays else None,
        protocol='threshold same-window scores; overlapping false windows merge; one distinct positive window per annotated event; decision at window end; no inference-time claim')


def evaluate(records, thresholds):
    labels = [r['label'] for r in records]
    result = {}
    for key, th in thresholds.items():
        result[key] = dict(threshold=th, clips=count_metrics(labels, [r['scores'][key] for r in records], th),
                           continuous=continuous_metrics(records, key, th))
    base, fused = result['legacy_gated']['clips'], result['three_stream']['clips']
    passed = fused['recall'][1] > base['recall'][1] and fused['normal_false_positive_rate'] < base['normal_false_positive_rate']
    return dict(simultaneously_improved=passed, variants=result, samples=len(records))


def verify_data_seal(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = read(manifest_path)
    sealed = read(manifest_path.with_name('manifest.seal.json'))
    if manifest.get('status') != 'sealed' or manifest.get('training_allowed') is not True:
        raise ValueError('Formal fusion requires a sealed training-ready manifest')
    if sealed['manifest_sha256'] != sha(manifest_path):
        raise ValueError('Data manifest changed after source isolation was sealed')
    rows = manifest['rows']
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source IDs')
    # Historical legacy conflicts are reported elsewhere; a fusion-fit or
    # calibration source may never overlap any branch or test source.
    for key in ('group', 'sha256'):
        buckets = {}
        for row in rows:
            buckets.setdefault(row[key], set()).add(row['split'])
        if any(len(splits) > 1 and splits & {'fusion_fit', 'calibration', 'test'} for splits in buckets.values()):
            raise ValueError(f'Independent fusion/calibration/test {key} leakage')
    return manifest


def verify_selection_seals(manifest_path, current_signature=None, current_seed=None):
    """All three decisions must be immutable before any new test prediction."""
    manifest_sha = sha(manifest_path)
    expected = dict(manifest_sha256=manifest_sha, baseline_config_sha256=sha(ROOT/'config/live_actions.json'),
                    fit_code_sha256=sha(__file__), feature_code_sha256=sha(ROOT/'scripts/fight_fusion_features.py'),
                    model_code_sha256=sha(ROOT/'backend/vision/fight_fusion.py'))
    selections = {}
    for seed in (42, 43, 44):
        folder = OUT/'fusion'/f'seed{seed}'
        path = folder/'selection_seal.json'
        if not path.is_file():
            raise ValueError(f'All three fusion selections must be sealed before testing; missing seed {seed}')
        record = read(path)
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Seed {seed} selection has stale data, baseline or code')
        if record.get('test_accessed_before_seal') is not False:
            raise ValueError('Selection cannot establish untouched test access')
        for spec in record['branches'].values():
            if sha(spec['path']) != spec['sha256']:
                raise ValueError('Sealed branch checkpoint changed')
        for variant, checksum in record['fusion_weights'].items():
            if sha(folder/(variant+'.pt')) != checksum:
                raise ValueError('Sealed fusion checkpoint changed')
        if seed == current_seed and record.get('signature') != current_signature:
            raise ValueError('Current evaluation does not match its sealed selection')
        selections[seed] = record
    return selections


def configure_numerical_policy():
    """Match branch training before pose extraction or cached-feature scoring."""
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    # Capture actual settings in the experiment signature, not just a prose
    # promise. No CUDA allocation is needed to inspect these backend flags.
    return dict(torch_threads=torch.get_num_threads(), opencv_threads=cv2.getNumThreads(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                pose_precision='FP32', branch_precision='CUDA FP16 autocast',
                fusion_precision='CPU FP32', new_sigmoid_precision='FP32',
                baseline_sigmoid_precision='original CUDA FP16')


def run(seed=42, phase='fit'):
    if phase not in ('fit', 'evaluate', 'all'):
        raise ValueError('Unknown fusion phase')
    offline()
    numerical_policy = configure_numerical_policy()
    manifest_path = OUT/'data/manifest.json'
    manifest = verify_data_seal(manifest_path)
    store = FeatureStore(manifest)
    specs, arm = branch_selection(seed, sha(manifest_path), store.signature)
    folder = OUT/'fusion'/f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    signature = digest(dict(manifest=sha(manifest_path), branches=specs, seed=seed,
        numerical_policy=numerical_policy,
        code=sha(__file__), feature_code=sha(ROOT/'scripts/fight_fusion_features.py'),
        model_code=sha(ROOT/'backend/vision/fight_fusion.py'), baseline_config=sha(ROOT/'config/live_actions.json')))
    # Check before constructing a scorer or entering any test-scoring path.
    decisions = verify_selection_seals(manifest_path, signature, seed) if phase == 'evaluate' else None
    scorer = CachedScorer(specs, store)
    try:
        if phase in ('fit', 'all'):
            fit_rows = [r for r in manifest['rows'] if r['split']=='fusion_fit']
            cal_rows = [r for r in manifest['rows'] if r['split']=='calibration']
            fit_records = score_rows(fit_rows, scorer, folder/'scores/fit', signature)
            models = {name: fit_logistic(fit_records, name, seed, folder, signature)
                      for name in ('global_roi', 'three_stream')}
            cal = apply_models(score_rows(cal_rows, scorer, folder/'scores/calibration', signature, True), models)
            if not cal:
                raise ValueError('Empty independent calibration sources')
            labels = [r['label'] for r in cal]
            old_scores = [r['scores']['legacy_gated'] for r in cal]
            selections = {key: dominant_threshold(labels, [r['scores'][key] for r in cal], old_scores, scorer.old_threshold)
                          for key in ('global', 'global_roi', 'three_stream')}
            thresholds = dict(legacy_gated=scorer.old_threshold, legacy_raw=scorer.old_threshold,
                              **{key: value['threshold'] for key,value in selections.items()})
            atomic_json(folder/'calibration.json', dict(selections=selections, thresholds=thresholds, signature=signature))
            selection = dict(signature=signature, manifest_sha256=sha(manifest_path), branches=specs,
                             numerical_policy=numerical_policy,
                             skeleton_arm=arm, thresholds=thresholds, fusion_weights={k:sha(folder/(k+'.pt')) for k in models},
                             baseline_config_sha256=sha(ROOT/'config/live_actions.json'), fit_code_sha256=sha(__file__),
                             feature_code_sha256=sha(ROOT/'scripts/fight_fusion_features.py'),
                             model_code_sha256=sha(ROOT/'backend/vision/fight_fusion.py'), test_accessed_before_seal=False)
            seal(folder/'selection_seal.json', selection)
            atomic_json(folder/'fit_complete.json', dict(status='complete', signature=signature, seed=seed,
                                                        test_accessed=False, phase='fit'))
            if phase == 'fit':
                return
            decisions = verify_selection_seals(manifest_path, signature, seed)
        else:
            selection = decisions[seed]
            thresholds = selection['thresholds']
            models = {}
            for variant in ('global_roi', 'three_stream'):
                saved = torch.load(folder/(variant+'.pt'), map_location='cpu', weights_only=True)
                if saved['signature'] != signature or tuple(saved['feature_names']) != FUSION_FEATURE_NAMES:
                    raise ValueError('Fusion checkpoint no longer matches sealed feature schema')
                model = QualityFusion()
                model.load_state_dict(saved['state_dict'])
                models[variant] = model.eval()
        # Nothing above uses test rows, labels for selection, or predictions.
        outputs = {}
        for split in ('test','legacy_test'):
            rows = [r for r in manifest['rows'] if r['split']==split]
            if not rows:
                raise ValueError('Missing required evaluation split: '+split)
            records = apply_models(score_rows(rows, scorer, folder/'scores'/split, signature, True), models)
            report = evaluate(records, thresholds)
            report['by_dataset'] = {d:evaluate([r for r in records if r['dataset']==d], thresholds)
                                    for d in sorted({r['dataset'] for r in records})}
            atomic_json(folder/(split+'_predictions.json'), records)
            atomic_json(folder/(split+'_metrics.json'), report)
            outputs[split] = report
        candidate = {k:{'path':v['path'],'sha256':v['sha256']} for k,v in specs.items()}
        # A threshold above one deliberately denotes no viable candidate. Do not
        # emit a deployable manifest in that case or pretend it is an improvement.
        th = thresholds['three_stream']
        if 0 <= th <= 1:
            candidate.update(fusion=dict(path=str(folder/'three_stream.pt'),sha256=sha(folder/'three_stream.pt')),
                threshold=th, device='cuda', max_rois=4,padding=.2,
                label='Three-stream classroom fight candidate',status='candidate_only',
                accepted_on_new_test=outputs['test']['simultaneously_improved'],
                pose=dict(path=str(ROOT/'models/yolov8n-pose.pt'),sha256=sha(ROOT/'models/yolov8n-pose.pt')),
                selection_seal=str(folder/'selection_seal.json'))
            atomic_json(folder/'candidate.json',candidate)
        lines = ['# Three-stream fight experiment', '',
                 'New held-out test passed both objectives: **'+str(outputs['test']['simultaneously_improved'])+'**.', '',
                 '| Variant | Recall | Precision | Normal clip FPR | Coverage |', '|---|---:|---:|---:|---:|']
        for name,v in outputs['test']['variants'].items():
            m=v['clips'];lines.append(f'| {name} | {m["recall"][1]:.4f} | {m["precision"][1]:.4f} | {m["normal_false_positive_rate"]:.4f} | {m["coverage"]:.4f} |')
        lines += ['', 'Production config was not changed. Legacy test has evaluation history. These are public-video results, not a measured classroom guarantee.',
                  'Legacy gated baseline replays original detector confidence and people guard on identical sampled windows. Full camera scheduling/view guards are evaluated separately.',
                  'Pose failures remain in all denominators. See predictions and continuous-event definitions in JSON.']
        (folder/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        atomic_json(folder/'complete.json',dict(status='complete',signature=signature,seed=seed,
                                              new_test_passed=outputs['test']['simultaneously_improved']))
    finally:
        store.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=42,choices=[42,43,44])
    parser.add_argument('--phase', choices=['fit', 'evaluate', 'all'], default='fit')
    args = parser.parse_args()
    run(args.seed, args.phase)
