"""Isolated script-dispatch experiment; not imported by the training queue."""
import weakref
import torch
from torch import nn
from scripts.stgcnpp_runtime_candidate import training_logit as transfer_candidate, bn_modules


class ScriptGCN(nn.Module):
    def __init__(self, source):
        super().__init__()
        assert source.adaptive == 'init' and source.conv_pos == 'pre' and source.with_res
        self.A = source.A
        self.num_subsets = source.num_subsets
        self.conv = source.conv
        self.bn = source.bn
        self.act = source.act
        self.down = source.down if isinstance(source.down, nn.Module) else nn.Identity()

    def forward(self, x):
        n, c, t, v = x.shape
        res = self.down(x)
        x = self.conv(x)
        x = x.view(n, self.num_subsets, -1, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A)).contiguous()
        return self.act(self.bn(x) + res)


class ScriptBlock(nn.Module):
    __constants__ = ['zero_residual']

    def __init__(self, source, index):
        super().__init__()
        self.gcn = ScriptGCN(source.gcn)
        self.tcn = source.tcn
        self.relu = source.relu
        self.zero_residual = index == 0
        self.residual = source.residual if isinstance(source.residual, nn.Module) else nn.Identity()

    def forward(self, x):
        if self.zero_residual:
            return self.relu(self.tcn(self.gcn(x)) + 0)
        res = self.residual(x)
        return self.relu(self.tcn(self.gcn(x)) + res)


class ScriptBackbone(nn.Module):
    def __init__(self, source):
        super().__init__()
        assert source.data_bn_type == 'VC'
        for module in source.modules():
            if isinstance(module, nn.Dropout):
                module.p = float(module.p)
        self.data_bn = source.data_bn
        self.gcn = nn.ModuleList([ScriptBlock(block, i) for i, block in enumerate(source.gcn)])

    def forward(self, x):
        n, m, t, v, c = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(n * m, v * c, t))
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
        for block in self.gcn:
            x = block(x)
        return x.reshape(n, m, x.size(1), x.size(2), x.size(3))


_CACHE = weakref.WeakKeyDictionary()


def training_logit(model, data):
    if model not in _CACHE:
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        compiled = torch.jit.script(ScriptBackbone(model.backbone)).train()
        original = dict(model.backbone.named_parameters())
        assert original.keys() == dict(compiled.named_parameters()).keys()
        assert all(parameter is original[name] for name, parameter in compiled.named_parameters())
        compiled_modules = dict(compiled.named_modules())
        pairs = [(module, compiled_modules[name.removeprefix('backbone.')])
                 for name, module in bn_modules(model) if name.startswith('backbone.')]
        _CACHE[model] = (compiled, pairs)
    compiled, pairs = _CACHE[model]

    def forward(x):
        for eager_bn, script_bn in pairs:
            for name, value in eager_bn._buffers.items():
                if value is not None:
                    setattr(script_bn, name, value)
        return compiled(x)

    previous = model.backbone.forward
    model.backbone.forward = forward
    try:
        return transfer_candidate(model, data)
    finally:
        model.backbone.forward = previous
