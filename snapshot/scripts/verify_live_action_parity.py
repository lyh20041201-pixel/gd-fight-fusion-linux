"""Compare deployed inference against saved validation scores, never tune models."""
from pathlib import Path
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,torch
from backend.vision.live_actions import LiveActionModels

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    models=LiveActionModels(ROOT/'config/live_actions.json');records=[]
    for dataset,kind,predictions in [
        ('fallvision','fall',ROOT/'results/video_events/skeleton_stgcnpp_ab/fallvision/B/seed_42/best_validation_predictions.json'),
        ('vfd','fight',ROOT/'results/video_events/skeleton_comparison_round2/vfd/rgb/seed_42/finetune_validation_predictions.json')]:
        rows=read(predictions)['rows'];threshold=models.manifest[kind]['threshold']
        # Include threshold neighbours, both classes, and unknown if present.
        picked=sorted([r for r in rows if r['score'] is not None],key=lambda r:abs(r['score']-threshold))[:4]
        for label in [0,1]:picked+=next(([r] for r in rows if r['label']==label and r['score'] is not None),[])
        picked+=next(([r] for r in rows if r['score'] is None),[])
        for row in {r['sample_id']:r for r in picked}.values():
            cache=torch.load(ROOT/f'datasets/video_events/skeleton_rebuild/{dataset}/{row["sample_id"]}.pt',map_location='cpu',weights_only=True)
            values=[models.score_fall_clip(c,cache['source_shape']) if kind=='fall' else models.score_rgb_clip(c['rgb']) for c in cache['clips']]
            known=[v for v in values if v is not None];actual=max(known) if known else None
            expected=row['score']
            passed=actual==expected
            record=dict(dataset=dataset,sample_id=row['sample_id'],expected=expected,actual=actual,exact=passed)
            records.append(record);print(json.dumps(record),flush=True)
    out=ROOT/'results/live_actions/runtime_validation';out.mkdir(parents=True,exist_ok=True)
    result=dict(status='passed' if all(r['exact'] for r in records) else 'failed',torch=torch.__version__,records=records)
    (out/'score_parity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert result['status']=='passed'

if __name__=='__main__':main()
