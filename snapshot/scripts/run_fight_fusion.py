"""One resumable GPU queue for all branches, calibration, seals and evaluation."""
from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
import argparse
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.fight_fusion_features import OUT,atomic_json,check_space
from scripts.skeleton_common import sha,digest

CODE_FILES=[
    'scripts/run_fight_fusion.py','scripts/train_fight_fusion.py',
    'scripts/fight_fusion_features.py','scripts/fit_fight_fusion.py',
    'backend/vision/fight_fusion.py','backend/vision/fight_fusion_live.py',
    'scripts/serve_fight_fusion_candidate.py',
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@contextmanager
def lock_queue():
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'queue.lock').open('a+b') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def save_status(**kwargs):
    atomic_json(OUT/'queue_status.json',dict(pid=os.getpid(),updated=time.time(),**kwargs))


def current_signature():
    return {name:sha(ROOT/name) for name in CODE_FILES}


def run_child(args,label,code):
    if current_signature()!=code:
        raise RuntimeError('Queue code changed; refusing mixed-version training')
    check_space()
    logs=OUT/'logs';logs.mkdir(parents=True,exist_ok=True)
    logpath=logs/(label+'.log')
    env=os.environ.copy()
    env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    env['PYTHONUNBUFFERED']='1'
    command=[sys.executable,'-u',*map(str,args)]
    with logpath.open('ab') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=0)
        while process.poll() is None:
            save_status(status='running',stage=label,child_pid=process.pid,log=str(logpath),command=command)
            time.sleep(5)
        if process.returncode:
            raise RuntimeError(f'{label} failed with exit code {process.returncode}; see {logpath}')


def select_skeleton():
    rows={arm:read(OUT/'branches'/arm/'seed42/selection.json') for arm in ('skeleton_random','skeleton_ntu')}
    def key(arm):
        m=rows[arm]['validation']
        return m['recall'][1],-m['normal_false_positive_rate'],m['macro_f1']
    arm=max(('skeleton_random','skeleton_ntu'),key=key)
    sources={name:dict(path=str(OUT/'branches'/name/'seed42/selection.json'),
                       sha256=sha(OUT/'branches'/name/'seed42/selection.json')) for name in rows}
    record=dict(arm=arm,seed=42,criterion='branch_val recall at <=5% FPR, then lower FPR, macro F1; random on tie',
                source_selections=sources,test_accessed=False)
    path=OUT/'branches/skeleton_arm_selection.json'
    if path.exists() and read(path)!=record:
        raise ValueError('Skeleton-arm decision differs from frozen selection')
    atomic_json(path,record)
    return arm


def summarize():
    records=[]
    lines=['# Three-stream fight training results','',
           'Production configuration has not been changed. New test and historical regression are reported separately.','',
           '| Seed | New-test recall | New-test normal FPR | Both objectives passed |',
           '|---|---:|---:|---|']
    for seed in (42,43,44):
        folder=OUT/'fusion'/f'seed{seed}'
        done=read(folder/'complete.json')
        data=read(folder/'test_metrics.json')
        m=data['variants']['three_stream']['clips']
        records.append(dict(seed=seed,recall=m['recall'][1],false_positive_rate=m['normal_false_positive_rate'],
                            both_improved=done['new_test_passed']))
        lines.append(f'| {seed} | {m["recall"][1]:.4f} | {m["normal_false_positive_rate"]:.4f} | {done["new_test_passed"]} |')
    lines += ['', 'Seed 42 is the predeclared candidate; seeds 43 and 44 measure variability. No test-best seed selection.',
              'A failing experiment remains available for diagnosis and is not promoted automatically.',
              'Public-video evaluation does not demonstrate real-classroom acceptance. See each seed report for coverage, baselines and continuous-event metrics.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic_json(OUT/'summary.json',dict(status='complete',seeds=records,
        all_seeds_improve_both=all(r['both_improved'] for r in records),candidate_seed=42,production_changed=False))


def run(wait_manifest=False):
    with lock_queue():
        code=current_signature()
        while not (OUT/'data/manifest.seal.json').exists():
            if not wait_manifest:
                raise RuntimeError('Complete source manifest not sealed yet')
            save_status(status='waiting_for_sealed_data',stage='data',code_signature=digest(code))
            time.sleep(10)
        manifest=OUT/'data/manifest.json'
        seal=read(OUT/'data/manifest.seal.json')
        if sha(manifest)!=seal['manifest_sha256'] or not read(manifest).get('training_allowed'):
            raise RuntimeError('Data seal invalid')
        # A completed smoke is required before starting any expensive formal fit.
        for branch in ('global','roi','skeleton_random','skeleton_ntu'):
            smoke=OUT/'smoke_outputs_v4/smoke'/branch/'seed42/selection.json'
            if not smoke.exists() or read(smoke).get('status')!='complete':
                raise RuntimeError(f'Real-video GPU smoke not complete: {branch}')
            if sha(smoke.with_name('selected_best.pt'))!=read(smoke)['sha256']:
                raise RuntimeError('Smoke checkpoint provenance mismatch: '+branch)
        protocol=dict(version=1,manifest_sha256=sha(manifest),code=code,
            seeds=[42,43,44],candidate_seed=42,all_test_access_after_all_selection_seals=True,
            baseline_config_sha256=sha(ROOT/'config/live_actions.json'),started=time.time())
        protocol_path=OUT/'queue_protocol.json'
        if protocol_path.exists():
            existing=read(protocol_path)
            for key in ('manifest_sha256','code','baseline_config_sha256'):
                if existing[key]!=protocol[key]:
                    raise RuntimeError('Queue resume provenance differs: '+key)
        else:atomic_json(protocol_path,protocol)
        for branch in ('global','roi','skeleton_random','skeleton_ntu'):
            run_child(['scripts/train_fight_fusion.py','--branch',branch,'--seed','42'],f'train_{branch}_42',code)
        arm=select_skeleton()
        for seed in (43,44):
            for branch in ('global','roi',arm):
                run_child(['scripts/train_fight_fusion.py','--branch',branch,'--seed',seed],f'train_{branch}_{seed}',code)
        for seed in (42,43,44):
            run_child(['scripts/fit_fight_fusion.py','--seed',seed,'--phase','fit'],f'fit_calibrate_{seed}',code)
        for seed in (42,43,44):
            if not (OUT/'fusion'/f'seed{seed}/selection_seal.json').exists():
                raise RuntimeError('All model and threshold selections must be sealed before test')
        for seed in (42,43,44):
            run_child(['scripts/fit_fight_fusion.py','--seed',seed,'--phase','evaluate'],f'evaluate_{seed}',code)
        if sha(ROOT/'config/live_actions.json')!=protocol['baseline_config_sha256']:
            raise RuntimeError('Production configuration changed during experiment')
        summarize()
        save_status(status='complete',stage='complete',report=str(OUT/'REPORT.md'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-manifest',action='store_true')
    parser.add_argument('--status',action='store_true')
    args=parser.parse_args()
    if args.status:
        for name in ('queue_status.json','feature_progress.json','fusion_progress.json','data/download_scfd.json','data/download_ubi.json','data/prepare_progress.json'):
            path=OUT/name
            if path.exists():print(name,json.dumps(read(path),ensure_ascii=False))
        return
    try:run(args.wait_manifest)
    except Exception as exc:
        save_status(status='attention_required',error=repr(exc))
        raise


if __name__=='__main__':main()
