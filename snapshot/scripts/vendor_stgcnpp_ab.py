"""Vendor the exact selected upstream classes, replacing only framework plumbing."""
import ast, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
COMMIT='f2bf3a6b08e2e8dec744692d64efdb187fd6719a'
SRC=ROOT/'third_party'/f'pyskl_{COMMIT}'
DEST=ROOT/'backend/vision/stgcnpp_backbone.py'
SPECS={
 'pyskl/utils/graph.py':['k_adjacency','edge2mat','normalize_digraph','get_hop_distance','Graph'],
 'pyskl/models/gcns/utils/init_func.py':['conv_branch_init','conv_init','bn_init'],
 'pyskl/models/gcns/utils/gcn.py':['unit_gcn'],
 'pyskl/models/gcns/utils/tcn.py':['unit_tcn','mstcn'],
 'pyskl/models/gcns/stgcn.py':['STGCNBlock','STGCN'],
}

def main():
    header='''"""ST-GCN++ core copied from PYSKL (Apache-2.0).
Upstream commit: f2bf3a6b08e2e8dec744692d64efdb187fd6719a.
Class/function bodies are verbatim; registry/import/download plumbing is replaced.
License and source hashes: third_party/STGCNPP_NOTICE.json and STGCNPP_LICENSE.
"""
import copy as cp
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
EPS=1e-4

def build_norm_layer(cfg, channels):
    cfg=dict(cfg); kind=cfg.pop('type'); requires_grad=cfg.pop('requires_grad',True)
    if kind != 'BN': raise ValueError('Only the pinned BN config is supported')
    layer=nn.BatchNorm2d(channels,**cfg)
    for p in layer.parameters():p.requires_grad=requires_grad
    return 'bn',layer

def build_activation_layer(cfg):
    cfg=dict(cfg); kind=cfg.pop('type')
    if kind != 'ReLU':raise ValueError('Only the pinned ReLU config is supported')
    return nn.ReLU(**cfg)

def cache_checkpoint(path):
    if not Path(path).is_file():raise ValueError('Local checkpoint required; auto-download disabled')
    return path

def load_checkpoint(model,path,strict=True):
    data=torch.load(cache_checkpoint(path),map_location='cpu',weights_only=True)
    model.load_state_dict(data.get('state_dict',data),strict=strict)

'''
    chunks=[header];records=[]
    for rel,names in SPECS.items():
        raw=(SRC/rel).read_text(encoding='utf-8');tree=ast.parse(raw)
        nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for name in names:
            code=ast.get_source_segment(raw,nodes[name])
            chunks.append(code+'\n\n')
            records.append(dict(source=rel,symbol=name,body_sha256=hashlib.sha256(code.encode()).hexdigest()))
    result=''.join(chunks)
    if DEST.exists() and DEST.read_text(encoding='utf-8')!=result:raise ValueError('Existing vendored file differs')
    DEST.write_text(result,encoding='utf-8')
    (ROOT/'third_party/STGCNPP_LICENSE').write_bytes((SRC/'LICENSE').read_bytes())
    (ROOT/'third_party/STGCNPP_NOTICE.json').write_text(json.dumps(dict(commit=COMMIT,upstream='https://github.com/kennymckormick/pyskl',
        file=str(DEST),file_sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),verbatim_bodies=records,
        changes=['framework imports/registry removed','BN/ReLU constructors use torch directly','checkpoint loader requires local file and weights_only=True']),indent=2),encoding='utf-8')
    print('VENDORED',DEST,len(result))

if __name__=='__main__':main()
