"""Combine unchanged successful predictions with explicitly audited decode retries."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,hashlib
from scripts.train_video_events import metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--recovery',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    original=json.loads(Path(a.original).read_text());recovery=json.loads(Path(a.recovery).read_text());manifest=json.loads(Path(a.manifest).read_text())
    if original['status']=='running' or recovery['status']!='complete':raise ValueError('Both inference runs must have finished; recovery must be complete')
    for key in ('checkpoint_sha256','threshold','labels','protocol'):
        if original[key]!=recovery[key]:raise ValueError('Incompatible recovery: '+key)
    failed={row['path'] for row in original['failures']};retried={row['path'] for row in recovery['predictions']}
    if failed!=retried:raise ValueError('Recovery must cover exactly the original failures')
    predictions=original['predictions']+recovery['predictions'];lookup={r['path']:r for r in predictions}
    if len(lookup)!=len(predictions) or set(lookup)!={r['path'] for r in manifest['rows']}:raise ValueError('Incomplete/duplicated coverage')
    for row in manifest['rows']:
        pred=lookup[row['path']]
        if pred['sha256']!=row['sha256'] or pred['label']!=row['label']:raise ValueError('Source/label changed')
    result=dict(original);result['predictions']=[lookup[r['path']] for r in manifest['rows']]
    result.update(status='complete',failures=[],expected_samples=len(manifest['rows']),
                  manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
                  duration_corrections=manifest.get('duration_corrections',[]),
                  recovery_provenance=dict(original_run=str(Path(a.original).resolve()),recovery_run=str(Path(a.recovery).resolve()),
                                           original_failures=original['failures'],recovered_count=len(retried)),
                  elapsed_seconds=original['elapsed_seconds']+recovery['elapsed_seconds'])
    result['metrics']=metrics([r['label'] for r in predictions],[r['prediction'] for r in predictions],2)
    result['metrics']['accuracy']=sum(r['correct'] for r in predictions)/len(predictions)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'evaluation.json').exists():raise ValueError('Use a fresh result')
    (out/'evaluation.json').write_text(json.dumps(result,indent=2))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(result['status'],len(predictions),result['metrics'])

if __name__=='__main__':main()
