"""Check real training-only input extremes before launching the twelve runs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.stgcnpp_ab import *

def main():
    setup_runtime();checks=[]
    for name,manifest in manifests().items():
        audited=[r for r in read(OUT/'audit'/f'inputs_{name}.json')['rows'] if r['split']=='train' and r['usable']]
        chosen={max(audited,key=lambda r:r['max_tracks'])['sample_id'],max(audited,key=lambda r:r['windows'])['sample_id']}
        chosen.update(r['sample_id'] for r in audited[:4])
        rows=[r for r in manifest['rows'] if r['sample_id'] in chosen]
        assert all(r['split']=='train' for r in rows)
        model,_=make_model(manifest['task'],'B',819);model=model.cuda().train();store=ABStore(name)
        torch.cuda.reset_peak_memory_stats();started=time.monotonic();windows=0;tracks=0
        for row in rows:
            model.zero_grad(set_to_none=True);data=store.get(row);windows+=len(data['items']);tracks+=sum(len(x['valid']) for x in data['items'])
            with torch.autocast('cuda'):
                value=training_logit(model,data);loss=torch.nn.functional.binary_cross_entropy_with_logits(value,torch.tensor(float(row['label']),device='cuda'))
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.cuda.synchronize()
        checks.append(dict(dataset=name,samples=[r['sample_id'] for r in rows],source_split='train',windows=windows,track_windows=tracks,
            seconds=time.monotonic()-started,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2))
        print(checks[-1],flush=True);del model;torch.cuda.empty_cache()
    write(OUT/'verification/training_input_smoke.json',dict(status='passed',checks=checks,test_samples_used=0))

if __name__=='__main__':main()
