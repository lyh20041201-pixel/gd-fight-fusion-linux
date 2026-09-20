"""Final result consistency check; does not train, tune or deploy a model."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.fine_actions_bc import *


def main():
    verify_seal();pre=read(OUT/'pretest_audit.json')
    assert sha(ROOT/'config/live_actions.json')==pre['baseline_config_sha256']
    assert sha(ROOT/'scripts/evaluate_fine_actions_external.py')==pre['additional_evaluation_code_sha256']['scripts/evaluate_fine_actions_external.py']
    selections=read(OUT/'all_selections_before_test.json');assert len(selections)==6
    for key,s in selections.items():
        arm,seed=key.split('_');assert sha(OUT/arm/f'seed_{seed}'/'best.pt')==s['checkpoint_sha256']
        assert s['test_accessed_during_selection'] is False
    gmd=read(OUT/'gmd_test_summary.json');ext=read(OUT/'external_fallvision_summary.json')
    assert set(gmd)==set(ext)=={'A',*selections}
    for name,m in gmd.items():
        assert m['tp']+m['fn']==17 and m['fp']+m['tn']==20
        result=read(OUT/'evaluation'/f'{name}_gmd_test.json')
        assert len(result['predictions'])==587 and len(m['videos'])==37
        assert OUT.joinpath('all_selections_before_test.json').stat().st_mtime<=OUT.joinpath('evaluation',f'{name}_gmd_test.json').stat().st_mtime
    for m in ext.values():assert m['tp']+m['fn']==69 and m['fp']+m['tn']==69
    report=dict(status='complete',models=6,epochs=read(OUT/'conclusion.json')['training_epochs'],
        gmd_test_videos=37,external_test_videos=138,local_invariant_tests='7 passed; no skips on this machine',
        test_portability_note='After the pre-test audit, only skip-if-local-fixture-missing decorators were added to the tests; the seven assertions reran and passed. Training/evaluation code unchanged.',
        unit_tests_current_sha256=sha(ROOT/'tests/test_fine_actions_bc.py'),
        browser_verification=['report tables rendered','seed selection updated score labels and graph',
            'normal source video played to its end: readyState 4, no media error, scores/action updated',
            'fall source and third seed selected: matching source frame and graph visible'],
        http_verification='range 206, HEAD, four non-allowlisted paths denied',
        live_configuration_unchanged=True,live_configuration_sha256=sha(ROOT/'config/live_actions.json'),
        report_sha256=sha(OUT/'REPORT.md'),viewer_sha256=sha(OUT/'index.html'),
        artifact_storage_bytes=sum(p.stat().st_size for root in [OUT,CACHE] for p in root.rglob('*') if p.is_file()),
        new_download_bytes=0,viewer_url='http://127.0.0.1:8011/')
    write(OUT/'completion.json',report)
    print('COMPLETE: 6 models; 37 GMD + 138 external videos; live model unchanged',flush=True)


if __name__=='__main__':main()
