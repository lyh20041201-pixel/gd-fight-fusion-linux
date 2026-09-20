"""Reject scripted execution unless exact parity passes. Does not train models."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import traceback
import time
from scripts import verify_stgcnpp_runtime_candidate as check
from scripts.stgcnpp_script_candidate import training_logit


def main():
    check.setup_runtime()
    check.candidate = training_logit
    check.DEST = check.OUT / 'verification/performance_script_candidate'
    check.DEST.mkdir(parents=True, exist_ok=True)
    prior = check.DEST / 'initial_parity.json'
    if prior.exists():
        check.write(check.DEST / f'initial_parity_previous_{time.time_ns()}.json', check.read(prior))
    try:
        check.check('fall', 'B', check.synthetic(count=1), 1)
        check.write(check.DEST / 'initial_parity.json', dict(status='passed', test_samples_used=0))
        print('INITIAL_SCRIPT_PARITY_PASSED', flush=True)
    except Exception:
        error = traceback.format_exc()
        check.write(check.DEST / 'initial_parity.json', dict(status='rejected', error=error, test_samples_used=0))
        print(error, flush=True)
        return
    check.main()
    for name in ['parity.json', 'result.json']:
        path = check.DEST / name
        result = check.read(path)
        result['script_candidate_sha256'] = check.sha(check.ROOT / 'scripts/stgcnpp_script_candidate.py')
        result['probe_sha256'] = check.sha(__file__)
        check.write(path, result)


if __name__ == '__main__':
    main()
