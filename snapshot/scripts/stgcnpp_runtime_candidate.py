"""Candidate execution-only optimization; not enabled in the live queue.

Math, window order, RNG operations, BN restoration and maximum/tie selection
remain the sealed implementation's operations. Transfers use the same CUDA
stream without per-tensor host waits. BN snapshots coalesce copies by dtype;
restoration still creates independent buffers to preserve autograd versions.
"""
from collections import OrderedDict
import math
import weakref

import torch

_BN_MODULES = weakref.WeakKeyDictionary()
INPUT_CACHE_BYTES = 16 * 1024 * 1024


def bn_modules(model):
    if model not in _BN_MODULES:
        _BN_MODULES[model] = tuple((name, module) for name, module in model.named_modules()
                                  if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))
    return _BN_MODULES[model]


def snapshot_bn(model):
    result = {name: {} for name, module in bn_modules(model)}
    groups = {}
    for name, module in bn_modules(model):
        for key, value in module._buffers.items():
            if value is not None:
                groups.setdefault((value.device, value.dtype), []).append((name, key, value))
    for rows in groups.values():
        flat = torch.cat([value.detach().reshape(-1) for name, key, value in rows])
        offset = 0
        for name, key, value in rows:
            result[name][key] = flat[offset:offset + value.numel()].view_as(value)
            offset += value.numel()
    return result


def restore_bn(model, state):
    for name, module in bn_modules(model):
        for key, value in state[name].items():
            setattr(module, key, value.clone())


def training_logit(model, data):
    if not data['usable']:
        return None
    items = data['items']
    cache = OrderedDict()
    cache_bytes = 0

    def window(index):
        nonlocal cache_bytes
        if index in cache:
            cache.move_to_end(index)
            return model(cache[index][0])
        # Fall's forward never reads pair geometry. Keep those unused tensors
        # on the CPU, while preserving every value and all dictionary keys.
        source = items[index]
        value = {key: (tensor if model.task == 'fall' and key not in ('features', 'valid')
                       else tensor.cuda(non_blocking=True)) for key, tensor in source.items()}
        size = sum(t.numel() * t.element_size() for t in value.values() if t.is_cuda)
        if size <= INPUT_CACHE_BYTES:
            while cache and (cache_bytes + size > INPUT_CACHE_BYTES or len(cache) >= 2):
                _, (_, prior) = cache.popitem(last=False)
                cache_bytes -= prior
            cache[index] = (value, size)
            cache_bytes += size
        return model(value)

    if len(items) == 1:
        return window(0)
    winners = []
    best = None
    with torch.no_grad(), torch.autocast('cuda', cache_enabled=False):
        for index, item in enumerate(items):
            before = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
            bn = snapshot_bn(model)
            value = window(index)
            if value is None:
                continue
            score = float(value)
            if not math.isfinite(score):
                raise ValueError('Nonfinite MIL score')
            if best is None or score > best:
                best = score
                winners = [(index, before, bn)]
            elif score == best:
                winners.append((index, before, bn))
    after = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
    after_bn = snapshot_bn(model)
    values = []
    try:
        for index, before, bn in winners:
            torch.set_rng_state(before[0])
            torch.cuda.set_rng_state_all(before[1])
            restore_bn(model, bn)
            values.append(window(index).float())
    finally:
        torch.set_rng_state(after[0])
        torch.cuda.set_rng_state_all(after[1])
        restore_bn(model, after_bn)
    return torch.stack(values).mean() if values else None
