"""Sequential offline action-model queue. Does not alter pose extraction."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,subprocess,time,msvcrt
from scripts.skeleton_common import ROOT,read,offline
from scripts.skeleton_io import write
from scripts.skeleton_round2 import OUT,MANIFESTS,require_source_seal


def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',choices=['vfd','fallvision'],required=True);a=p.parse_args()
    offline();OUT.mkdir(parents=True,exist_ok=True)
    names='_'.join(a.datasets)
    with (OUT/f'queue_train_{names}.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        for name in a.datasets:require_source_seal(MANIFESTS/f'{name}.json')
        jobs=[(name,mod,seed) for name in a.datasets for mod in ['skeleton','rgb'] for seed in [42,43,44]]
        state=dict(status='running',datasets=a.datasets,completed=0,total=len(jobs),jobs=[])
        for name,mod,seed in jobs:
            job=f'train_{name}_{mod}_{seed}';log=OUT/(job+'.log')
            state.update(current_model=f'{name}/{mod}',seed=seed,current_job=job,updated_at=time.time())
            write(OUT/f'queue_train_{names}.json',state);print('START',job,flush=True)
            with log.open('a',encoding='utf-8') as stream:
                result=subprocess.run([sys.executable,str(ROOT/'scripts/train_skeleton_round2.py'),'--dataset',name,'--modality',mod,'--seed',str(seed)],
                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            state['jobs'].append(dict(job=job,status='complete' if result.returncode==0 else 'failed',exit_code=result.returncode,log=str(log)))
            if result.returncode:
                state['status']='failed';write(OUT/f'queue_train_{names}.json',state)
                raise RuntimeError(f'{job} failed. See {log}; saved checkpoints retained.')
            state['completed']+=1;write(OUT/f'queue_train_{names}.json',state)
            print('COMPLETE',job,state['completed'],'/',len(jobs),flush=True)
        state.update(status='complete',current_model=None,seed=None);write(OUT/f'queue_train_{names}.json',state)
        print('QUEUE COMPLETE',names,flush=True)


if __name__=='__main__':main()
