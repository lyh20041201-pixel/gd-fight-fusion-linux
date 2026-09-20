"""Independently audit completed cloud A/B inputs, results and count arithmetic."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_qwen_blind_ab import OUT, SOURCE, MODEL_DIR, PROMPT, read, write, sha, validate_response


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    protocol = read(out / 'protocol.json')
    data = read(out / 'evaluation.json')
    assert data['summary']['status'] == 'complete'
    rows = read(out / 'manifest.local.json')['rows']
    assert len(rows) == 207 == len(data['rows'])
    ids = {r['opaque_id'] for r in rows}
    assert len(ids) == len(rows)
    assert {p.stem for p in (out / 'attempts').glob('*.json')} == ids
    assert {p.stem for p in (out / 'request_audits').glob('*.json')} == ids
    assert sha(SOURCE) == protocol['source_manifest_sha256']
    assert sha(MODEL_DIR / 'selected_best.pt') == protocol['checkpoint_sha256']
    assert sha(ROOT / 'config/live_actions.json') == protocol['local_config_sha256']
    assert sha(ROOT / 'scripts/evaluate_qwen_blind_ab.py') == protocol['code_sha256']
    frame_count = 0
    counts = {key: [[0, 0, 0], [0, 0, 0]] for key in ['A_local', 'B_local_and_qwen', 'Qwen_alone_diagnostic']}
    all_usage = True
    costs = 0.
    for local, result in zip(rows, data['rows'], strict=True):
        assert local['opaque_id'] == result['opaque_id']
        opaque = local['opaque_id']
        folder = out / 'blind_inputs' / opaque
        blind, audit, attempt = read(folder / 'input.json'), read(out / 'request_audits' / (opaque + '.json')), read(out / 'attempts' / (opaque + '.json'))
        assert sha(folder / 'input.json') == local['blind_input_sha256']
        assert audit['system_prompt'] == protocol['prompt'] == PROMPT
        user = json.loads(audit['user_text'])
        assert set(user) == {'frame_count', 'frame_times_seconds', 'audio_provided'}
        assert user['audio_provided'] is False
        assert user['frame_times_seconds'] == [f['time_seconds'] for f in blind['frames']]
        assert user['frame_count'] == len(blind['frames'])
        assert audit['video_frame_sha256'] == [f['sha256'] for f in blind['frames']]
        assert audit['temperature'] == protocol['temperature']
        assert audit['presence_penalty'] == protocol['presence_penalty']
        assert audit['response_format'] == protocol['response_format']
        assert audit['model'] == protocol['qwen_model']
        for f in blind['frames']:
            assert sha(folder / f['file']) == f['sha256']
            frame_count += 1
        parsed, error = validate_response(attempt['raw'], user['frame_count'])
        if attempt['status'] == 'ok':
            assert parsed == attempt['payload'] and error is None and attempt['finish_reason'] == 'stop'
        q = {'fight': 1, 'non_fight': 0, 'uncertain': -1}.get((parsed or {}).get('decision'), -1) if attempt['status'] == 'ok' else -1
        a = local['a_prediction']
        b = 0 if a == 0 else -1 if a == -1 else q
        assert (a, b, q) == (result['a_prediction'], result['b_prediction'], result['qwen_prediction'])
        assert b != 1 or (a == 1 and q == 1)
        for key, pred in [('A_local', a), ('B_local_and_qwen', b), ('Qwen_alone_diagnostic', q)]:
            counts[key][local['label']][2 if pred == -1 else pred] += 1
        usage = attempt.get('usage')
        if usage:
            cost = (usage['prompt_tokens'] * 2.2 + usage['completion_tokens'] * 13.3) / 1e6
            assert abs(cost - attempt['estimated_cost_cny']) < 1e-10
            costs += cost
        else:
            all_usage = False
            costs += attempt['reserved_cost_cny']
    for key, cm in counts.items():
        met = data['summary']['metrics'][key]
        assert met['confusion_matrix'] == cm
        assert met['fn'] == cm[1][0] + cm[1][2]
        assert met['tn'] == cm[0][0]
        assert met['unknown'] == cm[0][2] + cm[1][2]
        assert abs(met['recall'] - cm[1][1] / sum(cm[1])) < 1e-12
        assert abs(met['fpr'] - cm[0][1] / sum(cm[0])) < 1e-12
    assert abs(costs - data['summary']['accounted_cost_cny']) < 1e-10
    audit = dict(status='passed', videos=len(rows), original_frames_verified=frame_count,
                 request_allowlist_verified=True, identical_prompt_and_parameters=True,
                 labels_and_local_predictions_excluded_from_requests=True,
                 exact_one_attempt_per_formal_video=True,
                 independent_count_recalculation=True, positive_unknowns_counted_as_misses=True,
                 production_model_configuration_unchanged=True, all_usage_returned=all_usage,
                 formal_cost_cny=costs,
                 preflight_cost_cny=protocol.get('preflight', {}).get('estimated_cost_cny', 0),
                 total_estimated_cost_cny=costs + protocol.get('preflight', {}).get('estimated_cost_cny', 0))
    write(out / 'verification.json', audit)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
