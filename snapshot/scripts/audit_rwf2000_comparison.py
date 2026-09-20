"""Post-run audit: independent counts, brute-force validation threshold, provenance."""
from pathlib import Path
import csv
import json
import math
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_rwf2000_comparison import OUT, check_seal, completed_rows, read, write
from scripts.rwf2000_model import file_sha


def classify(value, threshold):
    return -1 if value is None else int(value >= threshold)


def independent_counts(rows, field, threshold):
    cm = [[0, 0, 0], [0, 0, 0]]
    for row in rows:
        p = classify(row[field], threshold)
        cm[row['label']][p if p >= 0 else 2] += 1
    tn, fp, un = cm[0]
    fn, tp, up = cm[1]
    div = lambda a, b: a / b if b else None
    return dict(tp=tp, fp=fp, tn=tn, fn=fn + up, unknown=un + up,
                precision=div(tp, tp + fp), recall=div(tp, sum(cm[1])),
                fpr=div(fp, sum(cm[0])), f1=div(2*tp, 2*tp + fp + fn + up),
                coverage=div(tn+fp+fn+tp, len(rows)), confusion_matrix=cm)


def brute_threshold(rows, field):
    scores = sorted({r[field] for r in rows if r[field] is not None})
    candidates = scores + [float(np.nextafter(max(scores, default=1.), np.inf))]
    choices = []
    for threshold in candidates:
        m = independent_counts(rows, field, threshold)
        if m['fp'] * 20 > sum(r['label'] == 0 for r in rows):
            continue
        normal_f1 = 2*m['tn'] / max(1, 2*m['tn'] + m['fp'] + m['confusion_matrix'][1][0] + m['confusion_matrix'][0][2])
        macro = (normal_f1 + (m['f1'] or 0)) / 2
        choices.append(((m['recall'], -m['fpr'], macro, threshold), threshold))
    return max(choices)[1]


def main():
    protocol = check_seal()
    calibration = read(OUT / 'calibration.json')
    evaluation = read(OUT / 'evaluation.json')
    validation, test = completed_rows('validation'), completed_rows('test')
    for field in ('rwf_raw', 'rwf_gated'):
        assert brute_threshold(validation, field) == calibration[field]['threshold']
    for row in validation:
        assert file_sha(OUT / 'validation/samples' / (row['sample_id'] + '.json')) == calibration['sample_files_sha256'][row['sample_id']]
    tests = [
        ('R3D-18 单独', 'r3d_raw', protocol['current_threshold']),
        ('现有方案：R3D-18＋人员筛选', 'r3d_gated', protocol['current_threshold']),
        ('公开FGN 默认阈值0.5', 'rwf_raw', .5),
        ('公开FGN 验证集校准阈值', 'rwf_raw', calibration['rwf_raw']['threshold']),
        ('公开FGN＋同一人员筛选，校准阈值', 'rwf_gated', calibration['rwf_gated']['threshold']),
    ]
    baseline_changes, errors, sources = [], [], []
    pairs = {}
    for split, rows in [('validation', validation), ('test', test)]:
        for name, field, threshold in tests:
            actual = independent_counts(rows, field, threshold)
            saved = evaluation['splits'][split]['metrics'][name]
            for key, value in actual.items():
                assert saved[key] == value, (split, name, key)
        for row in rows:
            assert file_sha(row['path']) == row['source_sha256']
            sources.append(row['sample_id'])
            old_raw = max(c['r3d_prior_score'] for c in row['windows'])
            old_gated = max((c['r3d_prior_score'] for c in row['windows'] if c['eligible']), default=None)
            for field, old in [('r3d_raw', old_raw), ('r3d_gated', old_gated)]:
                if classify(old, protocol['current_threshold']) != classify(row[field], protocol['current_threshold']):
                    baseline_changes.append(dict(sample_id=row['sample_id'], split=split, field=field,
                                                 historical=old, current=row[field]))
            for j, clip in enumerate(row['windows']):
                if clip['rwf_error']:
                    errors.append(dict(sample_id=row['sample_id'], split=split, window=j, error=clip['rwf_error']))
                else:
                    assert len(clip['rwf_input']['input_sha256']) == 64
                    assert len(clip['rwf_input']['native_frame_indices']) == 64
                    assert clip['rwf_score'] == clip['rwf_probabilities'][0]
        deltas = {}
        for title, base, candidate, threshold in [
                ('model_only', 'r3d_raw', 'rwf_raw', calibration['rwf_raw']['threshold']),
                ('same_person_gate', 'r3d_gated', 'rwf_gated', calibration['rwf_gated']['threshold'])]:
            counts = dict(new_true_positives=0, lost_true_positives=0, removed_false_positives=0, new_false_positives=0)
            for row in rows:
                a = classify(row[base], protocol['current_threshold'])
                b = classify(row[candidate], threshold)
                if row['label'] == 1:
                    counts['new_true_positives'] += a != 1 and b == 1
                    counts['lost_true_positives'] += a == 1 and b != 1
                else:
                    counts['removed_false_positives'] += a == 1 and b != 1
                    counts['new_false_positives'] += a != 1 and b == 1
            deltas[title] = counts
        pairs[split] = deltas
    audit = dict(passed=True, independent_metric_counts=True, brute_force_calibration_matches=True,
                 unchanged_runtime_config=True, source_hashes_checked=len(sources),
                 total_windows=sum(len(r['windows']) for r in validation + test),
                 baseline_classification_changes=baseline_changes, inference_errors=errors,
                 paired_changes=pairs, evaluation_sha256=file_sha(OUT / 'evaluation.json'),
                 calibration_sha256=file_sha(OUT / 'calibration.json'), protocol_sha256=file_sha(OUT / 'protocol.json'))
    write(OUT / 'audit.json', audit)
    # Concise, directly reviewable held-out cases, including both successes and regressions.
    lines = ['# 保留测试集逐视频复查', '', '标签沿用来源，0=非打架、1=打架；分歧清单不代表来源标签已重新确认。', '',
             '| 视频 | 来源标签 | 现有方案 | FGN＋同一人员筛选 |', '|---|---:|---|---|']
    changed = []
    names = {-1: '不确定', 0: '非打架', 1: '打架'}
    for row in test:
        a = classify(row['r3d_gated'], protocol['current_threshold'])
        b = classify(row['rwf_gated'], calibration['rwf_gated']['threshold'])
        if a != b:
            path = Path(row['path'])
            lines.append(f"| [{path.name}]({path.as_posix()}) | {row['label']} | {names[a]} | {names[b]} |")
            changed.append(dict(sample_id=row['sample_id'], label=row['label'], path=row['path'], current=a, rwf=b))
    (OUT / 'CASE_REVIEW.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write(OUT / 'disagreements.json', {'rows': changed})
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
