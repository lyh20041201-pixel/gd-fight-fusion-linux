import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from backend.vision import fight_fusion as model_api
from scripts import fight_fusion_features as features
from scripts import fit_fight_fusion as fit
from scripts.train_fight_fusion import window_logit


class TinyRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv3d(3, 2, 1)
        self.layer1 = self.layer2 = self.layer3 = nn.Identity()
        self.layer4 = nn.Conv3d(2, 2, 1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(2, 2)


def test_cached_cpu_head_finetune_and_direct_predictor_rgb_are_identical():
    torch.manual_seed(12)
    model = TinyRGB().eval()
    store = object.__new__(features.FeatureStore)
    store.device, store.extractor = torch.device('cpu'), model
    rgb = np.random.default_rng(1).integers(0, 256, size=(4, 13, 15, 3), dtype=np.uint8)
    layer3, pooled = store._features(rgb)
    scorer = object.__new__(fit.CachedScorer)
    scorer.device = store.device
    direct = model_api.rgb_logit(model, rgb)
    cached = scorer.rgb(model, layer3)
    window = dict(global_layer3=layer3, global_pooled=pooled)
    head = window_logit(model, window, 'global', 'head', store.device)
    fine = window_logit(model, window, 'global', 'finetune', store.device)
    assert layer3.dtype == torch.float32
    assert torch.equal(direct, cached) and torch.equal(cached, head) and torch.equal(head, fine)
    # Exercise backward, not just collection/inference: cache hits must contain
    # ordinary tensors acceptable to trainable modules.
    assert not torch.is_inference(layer3) and not torch.is_inference(pooled)
    head.backward()
    assert model.fc.weight.grad is not None


def test_lazy_model_construction_preserves_all_cpu_random_streams(monkeypatch):
    store = object.__new__(features.FeatureStore)
    store.device, store.pose, store.extractor = torch.device('cpu'), object(), None
    def construct():
        random.random()
        np.random.rand()
        return TinyRGB()
    monkeypatch.setattr(model_api, 'make_rgb_model', construct)
    random.seed(4)
    np.random.seed(5)
    torch.manual_seed(6)
    py, npstate, ts = random.getstate(), np.random.get_state(), torch.get_rng_state()
    store._init_models()
    assert random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], npstate[1])
    assert torch.equal(torch.get_rng_state(), ts)


def test_cache_budget_does_not_delete_everything_for_an_unstorable_window(tmp_path):
    store = object.__new__(features.FeatureStore)
    store.disk_limit, store.disk_bytes = 100, 20
    file = tmp_path/'cache.pt'
    file.write_bytes(b'x'*20)
    store.entries = {file: (20, 1.)}
    store.cache_root = tmp_path
    with pytest.raises(features.ResourcePause):
        store._evict(101)
    assert file.exists()


def test_frame_sampling_stays_bounded_and_rotates_without_using_test_labels():
    row = dict(sample_id='source', label_kind='frame', duration=3600,
               positive_intervals=[[400., 800.]])
    zero = features.selected_spans(row, epoch=0)
    one = features.selected_spans(row, epoch=1)
    assert len(zero) == len(one) == 16 and zero != one
    assert {features.window_label(row, s) for s in zero} == {0, 1}
    assert len(features.selected_spans(row, epoch=0, full=True)) > 1000
    assert features.all_spans(dict(sample_id='explicit', start=0., end=2.)) == [(0., 2.)]


def test_dataset_and_source_balancing_includes_normalization():
    records = [dict(dataset='a'), dict(dataset='a'), dict(dataset='b')]
    weights = fit.dataset_source_weights(records)
    assert weights.tolist() == [.25, .25, .5]
    size = len(model_api.FUSION_FEATURE_NAMES)
    xs = [torch.zeros(1, size), torch.full((100, size), 2.), torch.full((2, size), 10.)]
    model = model_api.QualityFusion()
    fit.normalize_dataset_sources(model, xs, weights)
    assert torch.allclose(model.feature_mean, torch.full((size,), 5.5))


def test_threshold_claim_requires_both_more_tp_and_fewer_fp():
    labels = [0, 0, 1, 1]
    result = fit.dominant_threshold(labels, [.1, .2, .8, .9], [.8, .2, .9, .1], .5)
    assert result['simultaneously_improved']
    assert result['metrics']['confusion_matrix'] == [[2, 0, 0], [0, 2, 0]]
    impossible = fit.dominant_threshold(labels, [.1, .2, .8, .9], [.1, .2, .9, .1], .5)
    assert not impossible['simultaneously_improved']  # zero old FPs cannot be reduced
    with pytest.raises(ValueError, match='matching'):
        fit.count_metrics(labels, [.1], .5)
    with pytest.raises(ValueError, match='finite probabilities'):
        fit.count_metrics([0], [float('nan')], .5)


def test_independent_data_seal_rejects_fit_source_leakage(tmp_path):
    rows = [dict(sample_id='a', group='same', sha256='one', split='train'),
            dict(sample_id='b', group='same', sha256='two', split='fusion_fit')]
    path = tmp_path/'manifest.json'
    features.atomic_json(path, dict(status='sealed', training_allowed=True, rows=rows))
    features.atomic_json(tmp_path/'manifest.seal.json', dict(manifest_sha256=features.sha(path)))
    with pytest.raises(ValueError, match='group leakage'):
        fit.verify_data_seal(path)
    rows[1]['group'] = 'other'
    features.atomic_json(path, dict(status='sealed', training_allowed=True, rows=rows))
    with pytest.raises(ValueError, match='changed'):
        fit.verify_data_seal(path)


def write_branch(root, branch, seed, recall=.7, smoke=False):
    folder = root/'branches'/branch/f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder/'selected_best.pt'
    checkpoint.write_bytes(b'synthetic sealed checkpoint')
    record = dict(status='complete', sha256=features.sha(checkpoint), selected_stage='head',
                  smoke=smoke, test_accessed=False,
                  validation=dict(recall=[1., recall], normal_false_positive_rate=.02, macro_f1=.8))
    features.atomic_json(folder/'selection.json', record)
    features.atomic_json(folder/'config.json', dict(smoke=smoke, manifest_sha256='manifest', feature_signature='features'))


def test_later_seeds_use_locked_arm_without_requiring_untrained_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(fit, 'OUT', tmp_path)
    for arm in ('skeleton_random', 'skeleton_ntu'):
        write_branch(tmp_path, arm, 42, .8 if arm == 'skeleton_random' else .6)
    sources = {arm: dict(path=str(tmp_path/'branches'/arm/'seed42/selection.json'),
                        sha256=features.sha(tmp_path/'branches'/arm/'seed42/selection.json'))
               for arm in ('skeleton_random', 'skeleton_ntu')}
    features.atomic_json(tmp_path/'branches/skeleton_arm_selection.json',
                         dict(seed=42, arm='skeleton_random', source_selections=sources))
    for branch in ('global', 'roi', 'skeleton_random'):
        write_branch(tmp_path, branch, 43)
    specs, arm = fit.branch_selection(43, 'manifest', 'features')
    assert arm == 'skeleton_random' and set(specs) == {'global', 'roi', 'skeleton'}
    write_branch(tmp_path, 'roi', 43, smoke=True)
    with pytest.raises(ValueError, match='Smoke'):
        fit.branch_selection(43, 'manifest', 'features')


def seal_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(fit, 'ROOT', tmp_path)
    monkeypatch.setattr(fit, 'OUT', tmp_path/'out')
    for name in ('config/live_actions.json', 'scripts/fight_fusion_features.py', 'backend/vision/fight_fusion.py'):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}', encoding='utf-8')
    manifest = tmp_path/'out/data/manifest.json'
    features.atomic_json(manifest, {})
    expected = dict(manifest_sha256=features.sha(manifest),
                    baseline_config_sha256=features.sha(tmp_path/'config/live_actions.json'),
                    fit_code_sha256=features.sha(fit.__file__),
                    feature_code_sha256=features.sha(tmp_path/'scripts/fight_fusion_features.py'),
                    model_code_sha256=features.sha(tmp_path/'backend/vision/fight_fusion.py'))
    return manifest, expected


def test_all_three_decisions_must_be_sealed_and_unchanged_before_testing(tmp_path, monkeypatch):
    manifest, expected = seal_environment(tmp_path, monkeypatch)
    def write(seed):
        folder = tmp_path/'out/fusion'/f'seed{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        model = folder/'three_stream.pt'
        model.write_bytes(b'sealed model')
        record = dict(expected, signature=f'sig{seed}', branches={}, test_accessed_before_seal=False,
                      fusion_weights={'three_stream': features.sha(model)})
        features.atomic_json(folder/'selection_seal.json', record)
    write(42)
    write(43)
    with pytest.raises(ValueError, match='missing seed 44'):
        fit.verify_selection_seals(manifest, 'sig42', 42)
    write(44)
    assert set(fit.verify_selection_seals(manifest, 'sig42', 42)) == {42, 43, 44}
    (tmp_path/'out/fusion/seed43/three_stream.pt').write_bytes(b'changed model')
    with pytest.raises(ValueError, match='checkpoint changed'):
        fit.verify_selection_seals(manifest, 'sig42', 42)


def test_fit_phase_never_scores_test_sources(tmp_path, monkeypatch):
    manifest_path, _ = seal_environment(tmp_path, monkeypatch)
    rows = [dict(label=y, split=split) for split in ('fusion_fit', 'calibration', 'test', 'legacy_test') for y in (0, 1)]
    monkeypatch.setattr(fit, 'offline', lambda: None)
    monkeypatch.setattr(fit, 'verify_data_seal', lambda _: dict(rows=rows))
    monkeypatch.setattr(fit, 'FeatureStore', lambda _: SimpleNamespace(signature='features', close=lambda: None))
    monkeypatch.setattr(fit, 'branch_selection', lambda *args: ({}, 'skeleton_random'))
    monkeypatch.setattr(fit, 'CachedScorer', lambda *args: SimpleNamespace(old_threshold=.5))
    observed = []
    def score(selected, *args, **kwargs):
        observed.extend(r['split'] for r in selected)
        assert all(r['split'] in ('fusion_fit', 'calibration') for r in selected)
        return selected
    monkeypatch.setattr(fit, 'score_rows', score)
    def train(records, variant, seed, folder, signature):
        (folder/(variant+'.pt')).write_bytes(b'fitted weights')
        return None
    monkeypatch.setattr(fit, 'fit_logistic', train)
    monkeypatch.setattr(fit, 'apply_models', lambda rs, ms: [dict(r, scores={k: float(r['label']) for k in
                                               ('legacy_gated', 'global', 'global_roi', 'three_stream')}) for r in rs])
    fit.run(42, 'fit')
    assert set(observed) == {'fusion_fit', 'calibration'}
    assert (tmp_path/'out/fusion/seed42/selection_seal.json').exists()
    assert not (tmp_path/'out/fusion/seed42/test_metrics.json').exists()


def test_baseline_uses_old_detector_threshold_and_deduplicated_observations(monkeypatch):
    from backend.vision import live_actions, action_guards
    clip = dict(keypoints=torch.zeros(0, 3, 17, 3), boxes=torch.zeros(0, 3, 4),
                timestamps=[0., 0., .1], frame_indices=[0, 0, 1])
    quality = model_api.window_quality(clip, (10, 10), [])
    scorer = object.__new__(fit.CachedScorer)
    scorer.device = torch.device('cpu')
    scorer.models = {'global': object(), 'roi': object(), 'skeleton': lambda _: None}
    scorer.old_models = [object()]
    scorer.old_config = dict(fight=dict(method='single'))
    calls = []
    class Pose:
        def predict(self, images, **kwargs):
            calls.append((len(images), kwargs))
            return [SimpleNamespace(keypoints=SimpleNamespace(data=torch.empty(0, 17, 3)),
                                    boxes=SimpleNamespace(xyxy=torch.empty(0, 4), conf=torch.empty(0))) for _ in images]
    scorer.store = SimpleNamespace(pose=Pose(), _init_models=lambda: None)
    monkeypatch.setattr(scorer, 'rgb', lambda *args: torch.tensor(1.))
    images = [np.zeros((10, 10, 3), np.uint8) for _ in range(3)]
    monkeypatch.setattr(fit, 'decode_window', lambda *args: (images, [0., 0., .1], [0, 0, 1], [10, 10]))
    monkeypatch.setattr(action_guards, 'fight_people_evidence', lambda _: {'eligible': False})
    window = dict(start=0., end=.1, label=0, quality=quality, global_layer3=torch.zeros(1),
                  roi_layer3=[], skeleton={})
    result = scorer.score(dict(sample_id='source'), window, include_baseline=True)
    assert calls[0][0] == 2 and calls[0][1]['conf'] == .25
    assert result['legacy_raw'] > .5 and result['legacy_gated'] is None


def test_logistic_fit_optimizes_real_cpu_objective_and_checks_resume_variant(tmp_path, monkeypatch):
    monkeypatch.setattr(fit, 'OUT', tmp_path)
    quality = dict.fromkeys(model_api.QUALITY_NAMES, 0.)
    records = [dict(label=label, label_kind='video', dataset=dataset,
                    windows=[dict(logits=[float(4*label-2), None, None], available=[True, False, False], quality=quality)])
               for dataset in ('a', 'b') for label in (0, 1)]
    model = fit.fit_logistic(records, 'three_stream', 42, tmp_path, 'signature')
    low, high = [float(model(fit.feature_matrix(record, 'three_stream')).detach()) for record in records[:2]]
    assert low < high
    saved = torch.load(tmp_path/'three_stream.pt', weights_only=True)
    assert saved['weighted_sampled_positive_prior'] == .5
    assert saved['loss_history'][-1] < saved['loss_history'][0]
    assert 'bias excluded' in saved['regularization']
    resumed = fit.fit_logistic(records, 'three_stream', 42, tmp_path, 'signature')
    assert torch.equal(model.linear.weight, resumed.linear.weight)
    saved['variant'] = 'global_roi'
    torch.save(saved, tmp_path/'three_stream.pt')
    with pytest.raises(ValueError, match='signature mismatch'):
        fit.fit_logistic(records, 'three_stream', 42, tmp_path, 'signature')


def temporal_record(duration=800., intervals=None, sample_id='events'):
    row = dict(sample_id=sample_id, label_kind='frame', duration=duration,
               positive_intervals=intervals if intervals is not None else [[20., 320.], [650., 650.2]])
    spans = features.all_spans(row)
    row['windows'] = [dict(start=a, end=b, label=features.window_label(row, (a, b))) for a, b in spans]
    return row, spans


def test_long_and_brief_events_receive_equal_sampling_mass_with_explicit_replacement():
    row, spans = temporal_record()
    counts = Counter()
    replacement_count = 0
    for epoch in range(12):
        indices, metadata = features.event_balanced_indices(spans, row['positive_intervals'], row['sample_id'],
                                                           epoch, 8, 8, return_metadata=True)
        assert len(indices) == 16
        counts.update(draw['event'] for draw in metadata['positive_draws'])
        replacement_count += metadata['replacement_draws']
        assert sum(row['windows'][i]['label'] for i in indices) == 8
        # All repeats are deliberate draws from the one group owning the window.
        for draw in metadata['positive_draws']:
            assert draw['group_events'] == metadata['window_event_groups'][draw['index']]
    assert counts == {0: 48, 1: 48}  # 300-second and .2-second events have equal mass
    assert replacement_count > 0


def test_cross_event_windows_have_exclusive_ownership_and_inseparable_groups_are_explicit():
    spans, events = [(0., 4.), (2., 6.), (4., 8.)], [[2.5, 3.], [3.2, 3.5]]
    indices, metadata = features.event_balanced_indices(spans, events, 'overlap', 0, 2, 0, return_metadata=True)
    assert len(set(indices)) == 2
    assert sorted(metadata['window_event_groups'].values()) == [[0], [1]]
    assert metadata['shared_event_groups'] == []
    # With one physical window, do not pretend two separately observed samples exist.
    indices, metadata = features.event_balanced_indices(spans[:1], events, 'inseparable', 0, 8, 0, return_metadata=True)
    assert indices == [0]
    assert metadata['shared_event_groups'] == [[0, 1]]
    assert metadata['window_event_groups'] == {0: [0, 1]}
    with pytest.raises(ValueError, match='unique before'):
        features.event_balanced_indices(spans[:1] * 2, events, 'duplicates', 0, 2, 0)


def test_event_rotation_is_deterministic_covers_more_events_than_budget_and_preserves_rng():
    intervals = [[float(10*i+2), float(10*i+3)] for i in range(13)]
    row, spans = temporal_record(140., intervals)
    counts = Counter()
    state = np.random.get_state()
    first = None
    for epoch in range(13):
        result = features.event_balanced_indices(spans, intervals, 'many', epoch, 4, 0, return_metadata=True)
        assert result == features.event_balanced_indices(spans, intervals, 'many', epoch, 4, 0, return_metadata=True)
        counts.update(d['event'] for d in result[1]['positive_draws'])
        if epoch == 0:
            first = result
        elif epoch == 1:
            assert result != first
    assert counts == {i: 4 for i in range(13)}
    assert np.array_equal(state[1], np.random.get_state()[1])


def test_feature_and_fit_sampling_share_event_semantics_and_evaluation_is_unchanged():
    row, spans = temporal_record()
    for epoch in (0, 3):
        wanted = features.event_balanced_indices(spans, row['positive_intervals'], row['sample_id'], epoch, 8, 8)
        assert features.selected_spans(row, epoch=epoch) == [spans[i] for i in wanted]
        indices, labels = fit.frame_choices(row, epoch)
        assert indices == features.event_balanced_indices(spans, row['positive_intervals'], row['sample_id'], epoch, 16, 16)
        assert len(indices) == 32 and sum(labels[indices]) == 16
    assert features.selected_spans(row, epoch=3, full=True) == spans
    assert features.selected_spans(row, epoch=None) == spans
    video_row = dict(row, label_kind='video')
    assert features.selected_spans(video_row, epoch=3) == spans


def test_event_sampling_handles_single_class_and_short_source_without_bypass():
    for intervals in ([], [[0., 10.]]):
        row, spans = temporal_record(10., intervals)
        selected = features.selected_spans(row, epoch=2)
        assert selected and len(selected) <= 16
        indices, labels = fit.frame_choices(row, 2)
        assert indices and set(labels[indices]) == ({1} if intervals else {0})
    row, spans = temporal_record(24., [[0., 14.], [23., 23.2]])
    assert len(spans) <= 16
    positive = sum(features.window_label(row, s) for s in spans)
    p = min(8, positive)
    n = min(16-p, len(spans)-positive)
    p = min(16-n, positive)
    desired = features.event_balanced_indices(spans, row['positive_intervals'], row['sample_id'], 0, p, n)
    assert features.selected_spans(row, epoch=0) == [spans[i] for i in desired]
    assert len(set(desired)) < len(desired)  # brief attack is deliberately revisited
