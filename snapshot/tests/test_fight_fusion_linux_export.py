"""Migration regressions: paths, checkpoint state and Linux resource limits."""
from pathlib import Path
import json
import pytest
import torch

from scripts.export_fight_fusion_linux import relative_source, migrate_checkpoint, signature
from scripts.linux_fight.guard import memory_available, PressureMonitor
from scripts.linux_fight.verify import contained


def test_windows_source_paths_portably_resolve_and_reject_escape(tmp_path):
    # Windows paths are parsed lexically even when this test runs on Linux.
    assert relative_source('datasets/clip.mp4', tmp_path) == 'datasets/clip.mp4'
    with pytest.raises(ValueError):
        relative_source('../outside.mp4', tmp_path)
    with pytest.raises(ValueError):
        contained(tmp_path, '../outside')


def test_checkpoint_migration_preserves_optimizer_rng_and_cursor(tmp_path):
    before = {'manifest': 'windows', 'stages': [['head', 10]]}
    after = {'manifest': 'linux', 'stages': [['head', 10]]}
    payload = dict(config={'manifest': 'windows', 'stages': [('head', 10)]}, config_signature=signature(before),
        model={'w': torch.tensor([1., 2.])},
        optimizer={'state': {0: {'exp_avg': torch.tensor([.1, .2]), 'step': torch.tensor(12.)}}},
        scaler={'scale': 1024.}, rng={'torch': torch.get_rng_state()},
        runner_state={'epochs_total': 8, 'cursor': 401, 'order': [3, 1, 2]})
    source, target = tmp_path / 'source.pt', tmp_path / 'copy.pt'
    torch.save(payload, source)
    original = source.read_bytes()
    report = migrate_checkpoint(source, target, before, after)
    result = torch.load(target, weights_only=True)
    assert report['state_verified_equal']
    assert source.read_bytes() == original
    assert result['config'] == after and result['config_signature'] == signature(after)
    assert result['runner_state']['cursor'] == 401
    assert torch.equal(result['optimizer']['state'][0]['exp_avg'], payload['optimizer']['state'][0]['exp_avg'])
    assert torch.equal(result['rng']['torch'], payload['rng']['torch'])


def test_migration_rejects_stale_source_signature(tmp_path):
    source = tmp_path / 'bad.pt'
    torch.save(dict(config_signature='changed', model={'w': torch.ones(1)}), source)
    with pytest.raises(ValueError, match='Source checkpoint configuration'):
        migrate_checkpoint(source, tmp_path / 'output.pt', {'version': 1}, {'version': 2})
    assert not (tmp_path / 'output.pt').exists()


def test_cgroup_limit_can_be_lower_than_host_free_memory(tmp_path):
    proc, groups = tmp_path / 'proc', tmp_path / 'cgroup'
    (proc / 'self').mkdir(parents=True)
    (groups / 'job').mkdir(parents=True)
    (proc / 'meminfo').write_text('MemAvailable: 32000000 kB\n')
    (proc / 'self/cgroup').write_text('0::/job\n')
    (groups / 'job/memory.max').write_text(str(12 * 1024**3))
    (groups / 'job/memory.current').write_text(str(7 * 1024**3))
    assert memory_available(proc, groups) == 5


def test_cgroup_ancestor_limit_is_enforced(tmp_path):
    proc, groups = tmp_path / 'proc', tmp_path / 'cgroup'
    (proc / 'self').mkdir(parents=True)
    (groups / 'job/child').mkdir(parents=True)
    (proc / 'meminfo').write_text('MemAvailable: 32000000 kB\n')
    (proc / 'self/cgroup').write_text('0::/job/child\n')
    (groups / 'job/memory.max').write_text(str(10 * 1024**3))
    (groups / 'job/memory.current').write_text(str(7 * 1024**3))
    (groups / 'job/child/memory.max').write_text('max')
    (groups / 'job/child/memory.current').write_text('0')
    assert memory_available(proc, groups) == 3


def test_linux_monitor_retains_soft_and_hard_stop_semantics():
    monitor = PressureMonitor()
    assert monitor.check(7, 1000, starting=True)
    monitor = PressureMonitor()
    assert monitor.check(5.9, 1000) is None
    assert monitor.check(5.9, 1000) is None
    assert monitor.check(5.9, 1000)
    assert PressureMonitor().check(3.9, 1000)
