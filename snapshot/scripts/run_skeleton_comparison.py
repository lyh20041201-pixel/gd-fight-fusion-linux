"""Sequential resumable job queue. Each job saves its own durable progress."""
from pathlib import Path
import sys,subprocess,time,argparse,msvcrt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.skeleton_common import ROOT,OUT,CACHE,write,read
from scripts.skeleton_io import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['train','external-extraction','evaluate','finalize'],required=True)
    p.add_argument('--wait-for-extraction',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/f'{a.phase}.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        jobs=[]
        if a.phase=='train':
            for name in ['gmd','tnue']:
                for modality in ['skeleton','rgb']:
                    for seed in [42,43,44]:
                        jobs.append((f'train_{name}_{modality}_{seed}',[str(ROOT/'scripts/train_skeleton_comparison.py'),'--dataset',name,'--modality',modality,'--seed',str(seed)],OUT/name/modality/f'seed_{seed}'/'selection.json'))
        elif a.phase=='external-extraction':
            for name in ['fallvision','vfd']:
                jobs.append((f'extract_{name}',[str(ROOT/'scripts/extract_skeleton_comparison.py'),'--dataset',name],OUT/f'extraction_{name}.json'))
        elif a.phase=='evaluate':
            for name in ['gmd','tnue']:
                jobs.append((f'evaluate_{name}',[str(ROOT/'scripts/evaluate_skeleton_comparison.py'),'--dataset',name],OUT/'evaluations'/name/'summary.json'))
        else:
            print('WAITING FOR both external evaluations',flush=True)
            while not all((OUT/'evaluations'/name/'summary.json').exists() for name in ['gmd','tnue']):
                state=read(OUT/'queue_evaluate.json') if (OUT/'queue_evaluate.json').exists() else {}
                if any(job.get('status')=='failed' for job in state.get('jobs',[])):
                    raise RuntimeError('Evaluation failed; finalization paused for repair')
                write(OUT/'queue_finalize.json',dict(phase='finalize',status='waiting_for_evaluation',updated_at=time.time()))
                time.sleep(15)
            jobs=[('build_acceptance',[str(ROOT/'scripts/build_skeleton_acceptance.py'),'build'],OUT/'acceptance/index.html'),
                  ('verify_complete',[str(ROOT/'scripts/verify_skeleton_comparison.py'),'--external'],OUT/'final_verification.json')]
        state=dict(phase=a.phase,status='running',jobs=[])
        for name,args,done in jobs:
            finished=done.exists() and (a.phase=='train' or
                (a.phase=='finalize' and (done.suffix=='.html' or read(done).get('status')=='passed')) or
                (a.phase not in ['train','finalize'] and read(done).get('status')=='complete'))
            if finished:
                state['jobs'].append(dict(job=name,status='already_complete'));continue
            if a.phase=='evaluate' and a.wait_for_extraction:
                ext='fallvision' if name=='evaluate_gmd' else 'vfd'
                progress=OUT/f'extraction_{ext}.json';expected=len(read(CACHE/'manifests'/f'{ext}.json')['rows'])
                print('WAITING FOR',ext,'extraction',flush=True)
                while True:
                    current=read(progress) if progress.exists() else {}
                    if current.get('completed',0)+len(current.get('failures',[]))==expected:break
                    write(OUT/f'queue_{a.phase}.json',dict(**state,waiting_for=ext,updated_at=time.time()))
                    time.sleep(15)
            write(OUT/f'queue_{a.phase}.json',dict(**state,current_job=name,updated_at=time.time()))
            print('START',name,flush=True)
            with (OUT/(name+'.log')).open('a',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state['jobs'].append(dict(job=name,status='complete' if result.returncode==0 else 'failed',exit_code=result.returncode))
            write(OUT/f'queue_{a.phase}.json',state)
            if result.returncode:raise RuntimeError(f'{name} failed; see its log. Completed artifacts retained for resume.')
        state['status']='complete';write(OUT/f'queue_{a.phase}.json',state);print('QUEUE COMPLETE',a.phase,flush=True)

if __name__=='__main__':main()
