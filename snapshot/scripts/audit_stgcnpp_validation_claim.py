"""Read-only independent audit of a reported validation score; no model inference."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import itertools
import json
import math
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/video_events/skeleton_stgcnpp_ab'
MANIFESTS = ROOT / 'datasets/video_events/skeleton_rebuild/round2/manifests'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def metric(rows, threshold):
    cm = [[0, 0, 0], [0, 0, 0]]
    for row in rows:
        pred = 2 if row['score'] is None else int(row['score'] >= threshold)
        cm[row['label']][pred] += 1
    precision = [cm[i][i] / max(1, sum(cm[j][i] for j in range(2))) for i in range(2)]
    recall = [cm[i][i] / max(1, sum(cm[i])) for i in range(2)]
    f1 = [2 * p * r / max(1e-12, p + r) for p, r in zip(precision, recall)]
    return dict(confusion_matrix=cm, precision=precision, recall=recall,
                macro_f1=sum(f1)/2, normal_false_positive_rate=cm[0][1]/sum(cm[0]),
                accuracy=(cm[0][0]+cm[1][1])/sum(map(sum, cm)),
                coverage=sum(sum(row[:2]) for row in cm)/sum(map(sum, cm)))


def main():
    started = time.perf_counter()
    results = {}
    for name in ['fallvision', 'vfd']:
        path = MANIFESTS / f'{name}.json'
        manifest = read(path)
        rows = manifest['rows']
        sealed = read(path.with_suffix('.seal.json'))
        assert sha(path) == sealed['manifest_sha256']
        assert digest([(r['sample_id'], r['label'], r['group'], r['split']) for r in rows]) == sealed['row_split_digest']
        assert len({r['sample_id'] for r in rows}) == len(rows)
        assert all(r['label'] == r['original_label'] and r['group'] == r['new_source_group'] for r in rows)
        evidence = {p: sha(p) == expected for p, expected in sealed['evidence_sha256'].items()}
        assert all(evidence.values())
        cross = {}
        for field in ['sample_id', 'sha256', 'group', 'path', 'canonical_source_basename']:
            if not all(field in r for r in rows):
                continue
            split_values = {s: {r[field] for r in rows if r['split'] == s} for s in ['train', 'validation', 'test']}
            overlaps = {f'{a}/{b}': len(split_values[a] & split_values[b])
                        for a, b in itertools.combinations(split_values, 2)}
            cross[field] = overlaps
            assert not any(overlaps.values()), (name, field, overlaps)
        results[name] = dict(
            rows=len(rows), recorded_identifiers_cross_split_overlap=cross,
            seals_and_evidence_verified=True,
            labels_unchanged_from_original_manifest=True,
            human_confirmed=sum(bool(r.get('human_confirmed')) for r in rows),
            split_summary={s: dict(rows=sum(r['split'] == s for r in rows),
                                  source_groups=dict(Counter(r['group'] for r in rows if r['split'] == s)),
                                  labels=dict(Counter(str(r['label']) for r in rows if r['split'] == s)))
                           for s in ['train', 'validation', 'test']},
            limitations=manifest['limitation'],
        )
        if name == 'fallvision':
            def check_source(row):
                return row['sample_id'], sha(row['path']) == row['sha256']
            failed = []
            with ThreadPoolExecutor(max_workers=2) as pool:
                for index, (sid, ok) in enumerate(pool.map(check_source, rows), 1):
                    if not ok:
                        failed.append(sid)
                    if index % 500 == 0 or index == len(rows):
                        print('REHASH_ORIGINAL_FALLVISION', index, '/', len(rows), flush=True)
            results[name]['original_video_SHA256_rechecked'] = len(rows)
            results[name]['changed_original_videos'] = failed
            assert not failed, failed
            val = {r['sample_id']: r for r in rows if r['split'] == 'validation'}

    folder = OUT / 'fallvision/B/seed_42'
    selection = read(folder / 'selection.json')
    predictions = read(folder / 'best_validation_predictions.json')['rows']
    assert len(predictions) == len(val) == len({r['sample_id'] for r in predictions})
    assert {r['sample_id'] for r in predictions} == set(val)
    assert all(r['label'] == val[r['sample_id']]['label'] for r in predictions)
    assert sha(folder / 'selected_best.pt') == selection['sha256']
    measured = metric(predictions, selection['threshold'])
    for key, value in measured.items():
        assert value == selection['validation'][key], (key, value, selection['validation'][key])
    finite = {float(r['score']) for r in predictions if r['score'] is not None}
    boundaries = sorted(finite | {math.nextafter(max(finite), math.inf)})
    candidates = []
    for threshold in boundaries:
        m = metric(predictions, threshold)
        if m['confusion_matrix'][0][1] * 20 <= sum(m['confusion_matrix'][0]):
            candidates.append(((m['recall'][1], -m['normal_false_positive_rate'], m['macro_f1'], threshold), threshold))
    assert max(candidates)[1] == selection['threshold']
    history = read(folder / 'run_record.json')['history']
    best = max(history, key=lambda h: (h['validation']['recall'][1], -h['validation']['normal_false_positive_rate'],
                                       h['validation']['macro_f1'], -h['epoch']))
    assert best['epoch'] == selection['epoch'] == 3
    result = dict(status='passed_with_material_scope_limits', datasets=results,
                  validation=dict(model='fallvision/B/seed_42', selected_epoch=selection['epoch'],
                                  metrics_recomputed_from_saved_scores=measured,
                                  threshold_independently_reselected=True, validation_sample_ids_and_labels_exact=True,
                                  selected_checkpoint_hash_verified=True),
                  current_AB_test_inference_access=False,
                  test_activity='Only provenance, recorded split metadata and file-hash checks; no test scores or models used',
                  label_audit_scope='Equality to original source labels; not an independent human action-label review',
                  duplication_audit_scope='Exact video bytes, IDs, paths, documented source groups and canonical basenames; does not rule out undiscovered re-edits or shared participants',
                  visual_spotcheck='Two existing scene contact sheets viewed as AI source evidence; not full independent relabeling',
                  not_a_new_blind_test=True, no_training_configuration_changed=True,
                  seconds=time.perf_counter()-started)
    dest = OUT / 'verification/validation_claim_audit.json'
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
