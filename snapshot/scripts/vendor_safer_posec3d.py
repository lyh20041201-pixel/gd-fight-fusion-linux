"""Copy pinned upstream model/heatmap bodies; replace MMCV construction only."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '900f0eba1e0abd55c290098fc6f221e7d5f8b6f9'
SOURCE = ROOT / 'third_party' / f'safer_pyskl_{COMMIT}'
SPECS = {
    'pyskl/models/cnns/resnet3d.py': ['BasicBlock3d', 'Bottleneck3d', 'ResNet3d'],
    'pyskl/models/cnns/resnet3d_slowonly.py': ['ResNet3dSlowOnly'],
    'pyskl/datasets/pipelines/augmentations.py': ['_combine_quadruple', 'PoseCompact'],
    'pyskl/datasets/pipelines/heatmap_related.py': ['GeneratePoseTarget'],
}
HEADER = '''"""Pinned SAFER/PYSKL PoseC3D inference core (Apache-2.0).
Source bodies retained; MMCV registry/import/construction replaced with PyTorch.
See third_party/SAFER_POSEC3D_NOTICE.json and SAFER_POSEC3D_LICENSE.
"""
import warnings
import numpy as np
import torch
from torch import nn
from torch.nn.modules.utils import _ntuple, _triple, _pair
from torch.nn.modules.batchnorm import _BatchNorm
EPS = 1e-3


def build_activation_layer(cfg):
    cfg = dict(cfg)
    if cfg.pop('type') != 'ReLU':
        raise ValueError('Only the pinned ReLU activation is supported')
    return nn.ReLU(**cfg)


class ConvModule(nn.Module):
    """Exact Conv3d -> BN3d -> ReLU subset used by the pinned checkpoint."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=False, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super().__init__()
        if conv_cfg != {'type': 'Conv3d'} or (norm_cfg or {}).get('type') != 'BN3d':
            raise ValueError('Unexpected upstream convolution configuration')
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        norm = dict(norm_cfg)
        norm.pop('type')
        requires_grad = norm.pop('requires_grad', True)
        self.bn = nn.BatchNorm3d(out_channels, **norm)
        for p in self.bn.parameters():
            p.requires_grad = requires_grad
        self.activate = build_activation_layer(act_cfg) if act_cfg else nn.Identity()

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


def _inference_only(*args, **kwargs):
    raise RuntimeError('Vendored inference core: use strict local checkpoint loading')


constant_init = kaiming_init = _load_checkpoint = load_checkpoint = cache_checkpoint = get_root_logger = _inference_only

'''


def main():
    chunks = [HEADER]
    records = []
    for rel, names in SPECS.items():
        raw = (SOURCE / rel).read_text(encoding='utf-8')
        nodes = {n.name: n for n in ast.parse(raw).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        for name in names:
            body = ast.get_source_segment(raw, nodes[name])
            chunks.append(body + '\n\n')
            records.append(dict(source=rel, symbol=name, sha256=hashlib.sha256(body.encode()).hexdigest()))
    path = ROOT / 'backend/vision/posec3d_core.py'
    path.write_text(''.join(chunks), encoding='utf-8')
    (ROOT / 'third_party/SAFER_POSEC3D_LICENSE').write_bytes((SOURCE / 'LICENSE').read_bytes())
    (ROOT / 'third_party/SAFER_POSEC3D_NOTICE.json').write_text(json.dumps(dict(
        upstream='https://github.com/safer-activities/pyskl', commit=COMMIT,
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), verbatim_bodies=records,
        changes=['MMCV ConvModule supported subset replaced with equivalent torch modules',
                 'Registries removed; training and downloading disabled']), indent=2), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
