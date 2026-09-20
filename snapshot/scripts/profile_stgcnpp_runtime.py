"""Bounded training-only runtime probe. Never changes the live experiment."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cProfile
import io
import json
import pstats
import time

from scripts.stgcnpp_ab import (
    OUT, ABStore, make_model, read, setup_runtime, torch, MANIFESTS,
    training_logit, write,
)


def main():
    setup_runtime()
    output = OUT / 'verification/performance_probe'
    output.mkdir(parents=True, exist_ok=True)
    model, _ = make_model('fall', 'B', 712)
    try:
        compiled = torch.jit.script(model.backbone)
        scripting = dict(supported=True)
        del compiled
    except Exception as exc:
        scripting = dict(supported=False, error=str(exc))
    write(output / 'script_feasibility.json', scripting)
    print('SCRIPT_FEASIBILITY', scripting, flush=True)
    model = model.cuda().train()
    manifest = read(MANIFESTS / 'fallvision.json')
    rows = [r for r in manifest['rows'] if r['split'] == 'train']
    audit = read(OUT / 'audit/inputs_fallvision.json')
    usable = {r['sample_id'] for r in audit['rows'] if r['split'] == 'train' and r['usable']}
    rows = sorted((r for r in rows if r['sample_id'] in usable), key=lambda r: r['sample_id'])[:12]
    store = ABStore('fallvision')
    criterion = torch.nn.BCEWithLogitsLoss()

    def step(row):
        data = store.get(row)
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            value = training_logit(model, data)
            loss = criterion(value, torch.tensor(float(row['label']), device='cuda'))
        loss.backward()

    for row in rows[:2]:
        step(row)
    torch.cuda.synchronize()
    profile = cProfile.Profile()
    started = time.perf_counter()
    profile.enable()
    for row in rows:
        step(row)
    torch.cuda.synchronize()
    profile.disable()
    elapsed = time.perf_counter() - started
    report = io.StringIO()
    pstats.Stats(profile, stream=report).sort_stats('cumulative').print_stats(35)
    (output / 'cpu_profile.txt').write_text(report.getvalue(), encoding='utf-8')
    result = dict(samples=len(rows), seconds=elapsed, seconds_per_sample=elapsed / len(rows),
                  training_samples_only=True, live_models_modified=False,
                  includes_optimizer=False, competing_live_models=2,
                  peak_cuda_MiB=torch.cuda.max_memory_allocated() / 2**20)
    write(output / 'summary.json', result)
    print(json.dumps(result), flush=True)
    print(report.getvalue(), flush=True)


if __name__ == '__main__':
    main()
