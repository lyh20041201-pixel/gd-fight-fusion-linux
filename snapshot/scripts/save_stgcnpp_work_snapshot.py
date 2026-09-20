"""Save an offline, streaming restart snapshot without changing experiment inputs."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save_json(path, value):
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    import psutil
    import torch
    from scripts.stgcnpp_ab import digest, model_config

    out = ROOT / 'results/video_events/skeleton_stgcnpp_ab'
    queue = json.loads((out / 'queue_status.json').read_text(encoding='utf-8'))
    assert queue['status'] == 'failed', 'Snapshot expects the already stopped queue'
    for job in queue.get('active_models', []):
        assert not psutil.pid_exists(job['pid']), 'Training process is still active'
    # Known workers from the stopped run must not still own the training command.
    for pid in (1117936, 1137616):
        if psutil.pid_exists(pid):
            assert 'scripts/train_stgcnpp_ab.py' not in psutil.Process(pid).cmdline()

    dest = ROOT / 'results/work_snapshots' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_before_restart')
    dest.mkdir(parents=True, exist_ok=False)
    save_json(dest / 'snapshot_status.json', {'status': 'saving', 'started_at': datetime.datetime.now().isoformat()})
    plan = json.loads((out / 'experiment_plan_frozen.json').read_text(encoding='utf-8'))
    for name, expected in plan['code_hashes'].items():
        assert sha(ROOT / name) == expected, 'Frozen code changed: ' + name
    for value in plan['manifests'].values():
        assert sha(Path(value['path'])) == value['sha256'], 'Sealed manifest changed'
    for asset in plan['download_receipt']['assets']:
        assert sha(Path(asset['path'])) == asset['sha256'], 'Pinned asset changed'

    progress = []
    for spec in plan['models']:
        folder = out / spec['dataset'] / spec['arm'] / ('seed_' + str(spec['seed']))
        row = dict(spec, status='not_started')
        signature = digest(model_config(spec['dataset'], spec['arm'], spec['seed']))
        if (folder / 'selection.json').exists():
            selection = json.loads((folder / 'selection.json').read_text(encoding='utf-8'))
            assert selection['config_signature'] == signature
            assert sha(folder / 'selected_best.pt') == selection['sha256']
            row.update(status='complete', epochs_trained=selection['epochs_trained'], selected_epoch=selection['epoch'])
        if (folder / 'resume.pt').exists():
            checkpoint = torch.load(folder / 'resume.pt', map_location='cpu', weights_only=True, mmap=True)
            assert checkpoint['config_signature'] == signature
            assert checkpoint['optimizer'] and checkpoint['rng']
            state = checkpoint['runner_state']
            row.update(checkpoint_sha256=sha(folder / 'resume.pt'), config_signature=signature,
                       completed_epochs=len(state['history']), next_epoch=state['epoch'] + 1,
                       cursor=state['cursor'], phase=state['phase'], optimizer_steps=state['optimizer_steps'],
                       rng_keys=list(checkpoint['rng']), optimizer_saved=True, scaler_saved=checkpoint['scaler'] is not None)
            if row['status'] != 'complete':
                row['status'] = 'interrupted_resumable'
            del checkpoint
        progress.append(row)
    save_json(dest / 'training_progress.json', progress)
    print('CHECKPOINTS_VERIFIED', sum(r['status'] == 'complete' for r in progress), 'complete; 2 resumable', flush=True)

    # Preserve large original data in place. Record existence, size, and prior audit hashes.
    refs = {}
    for path in sorted((out / 'audit').glob('inputs_*.json')):
        document = json.loads(path.read_text(encoding='utf-8'))
        for row in document['rows']:
            for field, hash_field in [('path', 'source_sha256'), ('raw_cache', 'raw_cache_sha256'), ('prepared_cache', 'prepared_cache_sha256')]:
                if row.get(field):
                    refs[row[field]] = {'recorded_sha256': row.get(hash_field), 'required': row.get(hash_field) is not None, 'kind': field}
    protected = json.loads((out / 'audit/protected_outputs.json').read_text(encoding='utf-8'))
    for row in protected['files']:
        refs.setdefault(row['path'], {'recorded_sha256': row['sha256'], 'required': True, 'kind': 'protected_prior_output', 'recorded_bytes': row.get('bytes')})
    missing = []
    with (dest / 'data_preserved_in_place.jsonl').open('w', encoding='utf-8', newline='\n') as handle:
        for name, prior in sorted(refs.items()):
            path = Path(name)
            exists = path.is_file()
            entry = dict(path=name, **prior, exists=exists, sha256_verification='prior audit hash retained; content not rehashed for this save operation')
            if exists:
                stat = path.stat()
                entry.update(current_bytes=stat.st_size, modified_ns=stat.st_mtime_ns)
            elif prior['required']:
                missing.append(name)
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    save_json(dest / 'data_presence_check.json', {'checked_paths': len(refs), 'missing_required_paths': missing,
              'original_data_copied': False, 'original_data_deleted_or_modified': False,
              'note': 'Presence/stat check plus retained historical SHA-256 records; not a fresh full-dataset hash audit.'})
    assert not missing, 'Required existing data files are missing; see data_presence_check.json'
    print('EXISTING_DATA_PATHS_VERIFIED', len(refs), flush=True)

    def git(*args):
        return subprocess.run(['git', *args], cwd=ROOT, check=True, stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW).stdout

    (dest / 'git_status_porcelain.txt').write_bytes(git('-c', 'core.quotepath=false', 'status', '--porcelain=v1'))
    (dest / 'uncommitted_changes.patch').write_bytes(git('diff', '--binary', 'HEAD'))
    (dest / 'git_head.txt').write_bytes(git('rev-parse', 'HEAD'))
    names = git('ls-files', '--cached', '--others', '--exclude-standard', '-z').decode('utf-8').split('\0')
    code_roots = {'backend', 'frontend', 'config', 'scripts', 'tests', 'docs', 'experiments', 'third_party'}
    paths = set()
    for name in names:
        if not name:
            continue
        relative = Path(name)
        path = ROOT / relative
        if path.is_file() and (len(relative.parts) == 1 or relative.parts[0] in code_roots):
            paths.add(path)
    for folder in [out, ROOT / 'datasets/video_events/skeleton_rebuild/round2/manifests',
                   ROOT / 'models/stgcnpp', ROOT / 'results/system_diagnostics']:
        paths.update(path for path in folder.rglob('*') if path.is_file() and path.suffix != '.lock')

    automation = Path('C:/Users/mcnuggets/.codex/automations/st-gcn-a-b/automation.toml')
    shutil.copy2(automation, dest / 'automation.toml')
    archive = dest / 'experiment_and_workspace.zip'
    index = []
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as bundle:
        for number, source in enumerate(sorted(paths), 1):
            before = source.stat()
            relative = source.relative_to(ROOT).as_posix()
            value = hashlib.sha256()
            with source.open('rb') as reader, bundle.open('workspace/' + relative, 'w', force_zip64=True) as writer:
                for block in iter(lambda: reader.read(1024 * 1024), b''):
                    value.update(block)
                    writer.write(block)
            after = source.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'File changed while being saved: ' + relative
            index.append({'path': relative, 'bytes': before.st_size, 'sha256': value.hexdigest()})
            if number % 200 == 0:
                print('SAVED_FILES', number, '/', len(paths), flush=True)
    with archive.open('rb+') as handle:
        handle.flush()
        os.fsync(handle.fileno())
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None, 'Backup archive CRC check failed'
    save_json(dest / 'snapshot_files.json', index)
    archive_hash = sha(archive)
    readme = '''# 重启后继续 ST-GCN++ A/B 实验

本次保存于 {saved_at}。训练已因 CUDA 内存不足停止，完成 2/12；此状态不是训练完成。

| 模型 | 已保存状态 |
|---|---|
| FallVision A，种子42 | 完成17轮，验证集选第9轮 |
| FallVision B，种子42 | 完成11轮，验证集选第3轮 |
| FallVision A，种子43 | 第16轮 cursor1478，前15轮完整保存 |
| FallVision B，种子43 | 第9轮 cursor3138，前8轮完整保存 |
| 其余8个模型 | 未开始 |

两路中断检查点均保存模型、优化器、混合精度缩放器、随机状态、顺序/游标、早停进度和配置签名。
原始视频、YOLO骨架/RGB缓存、旧轮次结果继续保留在 E:/GD 原位置。data_preserved_in_place.jsonl 记录现存路径和此前审计哈希；本次仅检查存在性，并未重新读取全部大型数据计算哈希。
experiment_and_workspace.zip 保存本次实验目录、封存清单、ST-GCN++预训练权重、项目代码/文档及未提交工作文件；snapshot_files.json 提供归档文件SHA-256。
这是一份同盘工作快照，不是原始数据的完整异盘备份。项目的既有删除/未提交修改记录在 git_status_porcelain.txt 和 uncommitted_changes.patch，本次没有提交或还原这些改动。

## 恢复步骤

1. 先保存其他软件的工作并重启/解决系统内存压力。此前实际约380个活动进程，却有约28.3万个内核进程对象，疑似资源未释放；责任程序/驱动尚未确定。诊断记录：E:/GD/results/system_diagnostics/memory_20260914_2004.json。
2. 确认本地数据、代码和检查点仍在，原训练进程已退出，内存有足够余量；先查看 `python scripts/status_stgcnpp_ab.py`。
3. 用户要求保持双路并行。在 E:/GD 使用原 Python 环境执行：

```powershell
Set-Location -LiteralPath 'E:/GD'
& 'C:/Users/mcnuggets/AppData/Local/Programs/Python/Python312/python.exe' -u scripts/run_stgcnpp_ab.py --max-workers 2 --oom-retries 2
```

队列核验并跳过已完成模型，其余从检查点续跑。不要重复启动，不要清零早停/训练轮次，不要用重新启动绕过持续未解决的OOM。封存核心代码、样本和配置不变；保持最多50轮、耐心8及验证集误报≤5%优先召回的选择规则。
12个模型全部选择封存后，队列才自动执行保留测试、原划分回归及验收。目前最终测试尚未开始；验证集100%召回不能当作测试成绩或教室泛化证据。
仍不下载新数据/模型/依赖，不重提取缓存，不改旧结果/报告/PPT，不做摄像头测试。

后续检查自动化 st-gcn-a-b 的配置另存 automation.toml；资源未改善时应保持停止，恢复须记录状态变化。运行环境需持续开机且不休眠。
'''.format(saved_at=datetime.datetime.now().isoformat(timespec='seconds'))
    with (dest / 'RESUME.md').open('w', encoding='utf-8', newline='\n') as handle:
        handle.write(readme)
        handle.flush()
        os.fsync(handle.fileno())
    status = {'status': 'saved_and_verified', 'saved_at': datetime.datetime.now().isoformat(),
              'directory': str(dest), 'archive': str(archive), 'archive_bytes': archive.stat().st_size,
              'archive_sha256': archive_hash, 'archived_files': len(index), 'preserved_data_paths_checked': len(refs),
              'missing_required_files': 0, 'archive_crc_verified': True, 'frozen_code_manifest_asset_hashes_verified': True,
              'completed_models': 2, 'interrupted_models_with_verified_checkpoint': 2, 'pending_models': 8,
              'training_restarted': False, 'original_files_modified_or_deleted': False}
    save_json(dest / 'snapshot_status.json', status)
    save_json(ROOT / 'results/work_snapshots/LATEST_STGCNPP_SAVE.json', status)
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
