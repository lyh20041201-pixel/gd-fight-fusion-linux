"""Read-only compact status; estimates use measured samples, never test scores."""
from pathlib import Path
import json,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/video_events/skeleton_stgcnpp_ab'

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    queue=read(OUT/'queue_status.json');runs=[]
    active={(r['dataset'],r['arm'],r['seed']) for r in queue.get('active_models',[])}
    for name in ['fallvision','vfd']:
        for arm in ['A','B']:
            for seed in [42,43,44]:
                folder=OUT/name/arm/f'seed_{seed}';p=folder/'progress.json'
                if not p.exists():continue
                item=read(p);record=folder/'run_record.json'
                if record.exists():
                    hist=read(record)['history'];item['epochs_finished']=len(hist)
                    if hist:item['measured_mean_epoch_seconds']=sum(h['seconds'] for h in hist)/len(hist)
                if item['phase']=='train' and item['completed']>=200:
                    item['estimated_training_seconds_per_epoch']=item['seconds']/item['completed']*item['total']
                    item['estimate_excludes_validation']=True
                if (folder/'selection.json').exists():
                    selection=read(folder/'selection.json');item['selected_epoch']=selection['epoch']
                item['queue_state']='complete' if (folder/'selection.json').exists() else 'active' if (name,arm,seed) in active else 'pending_resume' if queue.get('status')=='training' else 'interrupted'
                item['updated_seconds_ago']=time.time()-p.stat().st_mtime;runs.append(item)
    value=dict(queue=queue,runs=runs,finished_models=sum((OUT/n/a/f'seed_{s}'/'selection.json').exists() for n in ['fallvision','vfd'] for a in ['A','B'] for s in [42,43,44]),
        results_ready=(OUT/'verification/final.json').exists())
    print(json.dumps(value,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
