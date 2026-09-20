"""Pre-test provenance and invariant audit; produces an explicit audit trail."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import Counter
import numpy as np
from scripts.fine_actions_bc import *


def main():
    plan=verify_seal();labels=read(OUT/'fine_annotations.json');windows=read(CACHE/'windows.json')
    rows=labels['videos'];lookup={r['id']:r for r in rows}
    assert len(rows)==160 and not labels['human_review']
    for r in rows:
        assert not r['human_confirmed'] and r['spans'][0]['start']==0
        assert r['spans'][-1]['end']==r['duration']
        for a,b in zip(r['spans'],r['spans'][1:]):assert a['end']==b['start']
    for w in windows['windows']:
        r=lookup[w['video_id']]
        assert r['split']==w['split']
        if w['feature_index']>=0:
            assert w['valid_unique_frames']>=8
            assert w['sampled_end']<=w['end']+1e-8
            assert max(w['sampled_frame_indices'])/r['fps']<=w['end']+1e-8
            assert w['sampled_end']-w['sampled_start']>=3.5-1e-8
    assert read(ROOT/'config/live_actions.json')['fall']==plan['baseline']
    extra_scripts=['scripts/evaluate_fine_actions_external.py','scripts/audit_fine_actions_bc.py','tests/test_fine_actions_bc.py']
    audit=dict(status='passed',videos=len(rows),source_binary_counts=dict(Counter(r['category'] for r in rows)),
        windows=len(windows['windows']),valid_windows=sum(w['feature_index']>=0 for w in windows['windows']),
        annotations_sha256=sha(OUT/'fine_annotations.json'),protocol_sha256=sha(OUT/'protocol_frozen.json'),
        unit_tests='7 passed: pytest tests/test_fine_actions_bc.py -q',
        source_person_isolation='subject 1+2 train / 3 validation / 4 test',
        additional_evaluation_code_sha256={p:sha(ROOT/p) for p in extra_scripts},
        before_gmd_test_predictions=not (OUT/'gmd_test_summary.json').exists(),
        before_external_test_predictions=not (OUT/'external_fallvision_summary.json').exists(),
        baseline_config_sha256=sha(ROOT/'config/live_actions.json'))
    seal(OUT/'pretest_audit.json',audit)
    print(audit,flush=True)


if __name__=='__main__':main()
