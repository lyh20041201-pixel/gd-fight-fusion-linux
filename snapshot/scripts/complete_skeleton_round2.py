"""Monitor the two queues, then perform locked comparisons and build acceptance."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import subprocess,time,msvcrt
from scripts.skeleton_common import ROOT,read,seal,offline
from scripts.skeleton_io import write
from scripts.skeleton_round2 import OUT,MANIFESTS,require_source_seal

def run(command,log):
    with log.open('a',encoding='utf8') as f:
        result=subprocess.run([sys.executable,*command],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Process failed ({result.returncode}); {log}')

def snapshot():
    result=dict(phase='training',updated_at=time.time(),models_complete=0,models_total=12,queues={})
    for name in ['vfd','fallvision']:
        queue=read(OUT/f'queue_train_{name}.json');item=dict(status=queue['status'],completed=queue['completed'],total=6)
        if queue.get('current_model'):
            p=OUT/queue['current_model']/f'seed_{queue["seed"]}'/'progress.json'
            if p.exists():item['current']=read(p)
        result['queues'][name]=item
        for mod in ['skeleton','rgb']:
            for seed in [42,43,44]:
                p=OUT/name/mod/f'seed_{seed}'/'selection.json'
                if p.exists() and read(p)['status']=='complete':result['models_complete']+=1
    return result

def main():
    offline()
    with (OUT/'completion_pipeline.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        for name in ['fallvision','vfd']:require_source_seal(MANIFESTS/f'{name}.json')
        for name in ['source_gate','progress']:
            p=OUT/f'{name}.json';history=OUT/'audit'/f'{name}_before_training.json'
            if p.exists() and not history.exists():seal(history,read(p))
        write(OUT/'source_gate.json',dict(status='both_manifests_sealed_training_started',datasets={name:read(OUT/f'split_{name}.json') for name in ['fallvision','vfd']},
            source_review='AI scene/URL containment, not participant IDs or human confirmation',test_access='after both six-model queues complete',
            previous_audit_status=str(OUT/'audit/source_gate_before_training.json')))
        retried=set();print('MONITORING TWO TRAINING QUEUES',flush=True)
        while True:
            state=snapshot();write(OUT/'progress.json',state)
            statuses={name:q['status'] for name,q in state['queues'].items()}
            if all(v=='complete' for v in statuses.values()):break
            if 'running' not in statuses.values():
                failed=[name for name,v in statuses.items() if v=='failed']
                if not failed or any(name in retried for name in failed):raise RuntimeError('Training queue needs investigation: '+str(statuses))
                # One serial retry permits recovery after transient shared-device OOM
                # using the unchanged optimizer/RNG/config checkpoint.
                for name in failed:
                    retried.add(name);run(['scripts/run_skeleton_round2.py','--datasets',name],OUT/f'queue_{name}_recovery.log')
            time.sleep(30)
        if state['models_complete']!=12:raise ValueError('Training count mismatch')
        for name in ['vfd','fallvision']:
            write(OUT/'progress.json',dict(phase='evaluation',models_complete=12,models_total=12,dataset=name,updated_at=time.time()))
            run(['scripts/evaluate_skeleton_round2.py','--dataset',name],OUT/f'evaluate_{name}.log')
        write(OUT/'progress.json',dict(phase='acceptance',models_complete=12,models_total=12,updated_at=time.time()))
        run(['scripts/build_skeleton_round2_acceptance.py'],OUT/'build_acceptance.log')
        write(OUT/'progress.json',dict(phase='complete',models_complete=12,models_total=12,updated_at=time.time(),results=str(OUT/'ROUND2_RESULTS.md'),human_acceptance=False))
        print('ROUND2 TRAINING, EVALUATION AND ACCEPTANCE COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        write(OUT/'completion_pipeline_error.json',dict(status='needs_attention',error=str(e),updated_at=time.time()))
        raise
