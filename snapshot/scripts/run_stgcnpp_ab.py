"""Run the complete offline A/B queue, then frozen evaluation and acceptance."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,subprocess,msvcrt,time
from scripts.stgcnpp_ab import *

def command(args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as output:
        output.write('\nRUN '+time.strftime('%Y-%m-%d %H:%M:%S')+' '+str(args)+'\n');output.flush()
        child=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        while child.poll() is None:
            time.sleep(5)
        if child.returncode:raise RuntimeError(f'Command failed ({child.returncode}): {args}; log={log}')

def main(max_workers=2,oom_retries=2):
    setup_runtime();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        prep=read(OUT/'audit/preparation_complete.json');assert prep['status']=='complete'
        preflight=read(OUT/'verification/preflight.json');assert preflight['status']=='passed'
        frozen=read(OUT/'experiment_plan.json');frozen['code_hashes']={f:sha(ROOT/f) for f in CODE}
        frozen['phase']='frozen_after_preflight_before_any_AB_training'
        frozen['preflight_sha256']=sha(OUT/'verification/preflight.json')
        frozen['preparation_sha256']=sha(OUT/'audit/preparation_complete.json')
        frozen['execution']='At most two independent training processes on the local GPU; peak-memory smoke passed; each process has its own RNG/checkpoints'
        frozen['training_smoke_sha256']=sha(OUT/'verification/training_input_smoke.json')
        assert preflight['code_hashes']==frozen['code_hashes']
        seal(OUT/'experiment_plan_frozen.json',frozen)
        queue=[(n,a,s) for n in POLICY['datasets'] for s in POLICY['seeds'] for a in POLICY['arms']]
        start=time.monotonic()
        try:
            pending=[];active=[];completed=[];retries={}
            # Completed models are verified, counted, and never launched again.
            for name,arm,seed in queue:
                folder=OUT/name/arm/f'seed_{seed}'
                if (folder/'selection.json').exists():
                    saved=read(folder/'selection.json')
                    assert saved['status']=='complete' and saved['config_signature']==digest(model_config(name,arm,seed))
                    assert saved['sha256']==sha(folder/'selected_best.pt')
                    completed.append(dict(dataset=name,arm=arm,seed=seed,model='ST-GCN++',previously_completed=True))
                else:pending.append((name,arm,seed))
            execution=dict(started_at=time.strftime('%Y-%m-%d %H:%M:%S'),max_workers=max_workers,oom_retries=oom_retries,
                runner_sha256=sha(__file__),frozen_plan_sha256=sha(OUT/'experiment_plan_frozen.json'),
                previously_completed=len(completed),training_policy_changed=False)
            write(OUT/'recoveries'/f'execution_{time.time_ns()}.json',execution)
            try:
                while pending or active:
                    while pending and len(active)<max_workers:
                        name,arm,seed=pending.pop(0);log=OUT/'logs'/f'train_{name}_{arm}_{seed}.log';log.parent.mkdir(parents=True,exist_ok=True)
                        handle=log.open('a',encoding='utf-8');handle.write('\nRUN '+time.strftime('%Y-%m-%d %H:%M:%S')+'\n');handle.flush()
                        child=subprocess.Popen([sys.executable,'-u','scripts/train_stgcnpp_ab.py','--dataset',name,'--arm',arm,'--seed',str(seed)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                        info=dict(dataset=name,arm=arm,seed=seed,model='ST-GCN++',pid=child.pid,progress_file=str(OUT/name/arm/f'seed_{seed}'/'progress.json'))
                        active.append((child,handle,info));print('START_MODEL',info,flush=True)
                    for child,handle,info in list(active):
                        code=child.poll()
                        if code is not None:
                            handle.close();active.remove((child,handle,info))
                            if code:
                                job=(info['dataset'],info['arm'],info['seed'])
                                log=OUT/'logs'/f'train_{job[0]}_{job[1]}_{job[2]}.log'
                                tail=log.read_text(encoding='utf-8',errors='replace').rsplit('\nRUN ',1)[-1][-16000:]
                                oom='CUDA' in tail and 'out of memory' in tail.lower()
                                attempts=retries.get(job,0)
                                if oom and attempts<oom_retries:
                                    retries[job]=attempts+1
                                    pending.insert(0,job)
                                    event=dict(at=time.strftime('%Y-%m-%d %H:%M:%S'),job=list(job),reason='CUDA out of memory',
                                        retry=attempts+1,max_workers=max_workers,checkpoint_resume=True)
                                    write(OUT/'recoveries'/f'oom_retry_{time.time_ns()}.json',event)
                                    print('OOM_RETRY_FROM_CHECKPOINT',event,flush=True)
                                    # Retry from the checkpoint without changing requested concurrency.
                                    time.sleep(min(15*(attempts+1),45))
                                    continue
                                raise RuntimeError(f'Training failed: {info}; exit={code}')
                            completed.append(info);print('FINISH_MODEL',info,flush=True)
                    value=dict(status='training',completed=len(completed),total=12,active_models=[x[2] for x in active],pending=len(pending),seconds=time.monotonic()-start,
                        max_workers=max_workers,execution_started_at=execution['started_at'],seconds_scope='current queue invocation; per-model histories preserve earlier training')
                    write(OUT/'queue_status.json',value)
                    if active:time.sleep(5)
            finally:
                for child,handle,info in active:
                    if child.poll() is None:child.terminate();child.wait(timeout=30)
                    handle.close()
            write(OUT/'queue_status.json',dict(status='evaluation',completed=12,total=12,seconds=time.monotonic()-start))
            command(['scripts/evaluate_stgcnpp_ab.py'],OUT/'logs/evaluate.log')
            write(OUT/'queue_status.json',dict(status='acceptance',completed=12,total=12))
            command(['scripts/complete_stgcnpp_ab.py'],OUT/'logs/complete.log')
            write(OUT/'queue_status.json',dict(status='complete',completed=12,total=12,seconds=time.monotonic()-start,finished_at=time.strftime('%Y-%m-%d %H:%M:%S')))
            print('ALL_AB_COMPLETE',flush=True)
        except Exception as exc:
            write(OUT/'queue_status.json',dict(status='failed',error=str(exc),recoverable_by_same_command=True));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-workers',type=int,choices=[1,2],default=2)
    parser.add_argument('--oom-retries',type=int,choices=range(0,4),default=2)
    args=parser.parse_args();main(args.max_workers,args.oom_retries)
