"""Exact execution parity and interleaved timing on training-only inputs."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import statistics
import time

from scripts.stgcnpp_ab import *
from scripts.verify_stgcnpp_ab import synthetic
from scripts.stgcnpp_runtime_candidate import training_logit as candidate

DEST = OUT / 'verification/performance_candidate'


def assert_nested(a, b, path='state'):
    if torch.is_tensor(a):
        if not torch.equal(a, b):
            raise AssertionError(f'{path}: tensor differs; max diff {float((a-b).abs().max())}')
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_nested(a[key], b[key], f'{path}.{key}')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b)):
            assert_nested(x, y, f'{path}[{index}]')
    else:
        assert a == b, path


def train_step(model, optimizer, scaler, dataset, fn):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for data, label in dataset:
        with torch.autocast('cuda'):
            value = fn(model, data)
            if value is None:
                losses.append(None)
                continue
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                value, torch.tensor(float(label), device='cuda'))
        scaler.scale(loss / len(dataset)).backward()
        losses.append(float(loss))
    gradients = {n: None if p.grad is None else p.grad.detach().cpu().clone()
                 for n, p in model.named_parameters()}
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
    scaler.step(optimizer)
    scaler.update()
    return losses, gradients


def check(task, arm, data, label):
    a, _ = make_model(task, arm, 814)
    a = a.cuda().train()
    b = copy.deepcopy(a)
    opt_a = torch.optim.AdamW(a.parameters(), lr=.001, weight_decay=.0001)
    opt_b = torch.optim.AdamW(b.parameters(), lr=.001, weight_decay=.0001)
    sa = torch.amp.GradScaler('cuda')
    sb = torch.amp.GradScaler('cuda')
    batch = [(data, label)] * 4
    for step in range(2):
        before = rng_state()
        result_a = train_step(a, opt_a, sa, batch, training_logit)
        after = rng_state()
        restore_rng(before)
        result_b = train_step(b, opt_b, sb, batch, candidate)
        assert_nested(result_a, result_b, f'{task}/{arm}/step{step}/gradients_and_losses')
        assert_nested(a.state_dict(), b.state_dict(), 'all_parameters_and_BN')
        assert_nested(opt_a.state_dict(), opt_b.state_dict(), 'optimizer')
        assert_nested(sa.state_dict(), sb.state_dict(), 'scaler')
        assert_nested(after, rng_state(), 'RNG')
    a.eval()
    b.eval()
    assert score_sample(a, data) == score_sample(b, data)
    del a, b, opt_a, opt_b
    torch.cuda.empty_cache()


def timings():
    result = []
    for name, manifest in manifests().items():
        audit = read(OUT / 'audit' / f'inputs_{name}.json')['rows']
        eligible_ids = {r['sample_id'] for r in audit if r['split'] == 'train' and r['usable']}
        rows = sorted((r for r in manifest['rows'] if r['split'] == 'train' and r['sample_id'] in eligible_ids),
                      key=lambda r: digest(r['sample_id']))[:24]
        store = ABStore(name)
        datasets = [(store.get(row), row['label']) for row in rows]
        assert tensor_bytes(datasets) < 64 * 1024 * 1024
        model, _ = make_model(manifest['task'], 'B', 715)
        model = model.cuda().train()

        def run(fn, selected):
            for data, label in selected:
                model.zero_grad(set_to_none=True)
                with torch.autocast('cuda'):
                    value = fn(model, data)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        value, torch.tensor(float(label), device='cuda'))
                loss.backward()
            torch.cuda.synchronize()

        run(training_logit, datasets[:2])
        run(candidate, datasets[:2])
        repeats = []
        for repeat in range(4):
            order = [('base', training_logit), ('candidate', candidate)]
            if repeat % 2:
                order.reverse()
            row = {}
            for label, fn in order:
                before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                random_before = rng_state()
                started = time.perf_counter()
                run(fn, datasets)
                row[label] = time.perf_counter() - started
                model.load_state_dict(before)
                restore_rng(random_before)
            repeats.append(row)
            print('TIMING', name, repeat, row, flush=True)
        base = statistics.median(r['base'] for r in repeats)
        fast = statistics.median(r['candidate'] for r in repeats)
        result.append(dict(dataset=name, samples=len(rows), rows=[r['sample_id'] for r in rows],
                           repeats=repeats, median_base_seconds=base, median_candidate_seconds=fast,
                           median_speedup=base / fast, includes_data_loading=False,
                           includes_optimizer=False, other_live_training_processes=2))
        del model, datasets
        torch.cuda.empty_cache()
    return result


def main():
    setup_runtime()
    DEST.mkdir(parents=True, exist_ok=True)
    checked = []
    for task in ['fall', 'fight']:
        for arm in ['A', 'B']:
            for count, same in [(1, False), (3, False), (3, True)]:
                data = synthetic(count=count, same=same)
                check(task, arm, data, 1)
                row = dict(task=task, arm=arm, windows=count, tied=same)
                checked.append(row)
                print('EXACT_PARITY', row, flush=True)
                write(DEST / 'progress.json', dict(phase='exact_parity', completed=len(checked), total=12))
    result = dict(status='parity_passed', checks=checked, all_state_tensors_exact=True,
                  gradients_exact=True, optimizer_scaler_RNG_exact=True, test_samples_used=0,
                  candidate_sha256=sha(ROOT / 'scripts/stgcnpp_runtime_candidate.py'),
                  verifier_sha256=sha(__file__), core_hashes={f: sha(ROOT / f) for f in CODE})
    write(DEST / 'parity.json', result)
    result['timings'] = timings()
    result['status'] = 'complete'
    result['eligible_for_rollout'] = all(r['median_speedup'] >= 1.10 for r in result['timings'])
    write(DEST / 'result.json', result)
    print('CANDIDATE_RESULT', result, flush=True)


if __name__ == '__main__':
    main()
