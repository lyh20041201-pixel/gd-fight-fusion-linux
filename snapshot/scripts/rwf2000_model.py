"""Inference-only adapter for the author-published RWF repository FGN checkpoint.

The legacy HDF5 is inspected as data, never deserialized as executable Lambda
bytecode. All learned tensors are loaded verbatim into the same Keras graph.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / 'models/rwf2000/keras_model.h5'
EXPECTED_SHA = '46c0e07b83a1bea4a9390dbcf01352b095ce0fadf9e43719457e7beef085e473'
SOURCE = ROOT / 'third_party/rwf2000_1267a1208574af651dc2ea9b1439cb60b5a9fc54'


def file_sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_model(device='cuda'):
    import h5py
    import keras
    import torch
    if file_sha(WEIGHTS) != EXPECTED_SHA:
        raise ValueError('RWF checkpoint hash mismatch')
    tensors, learned = {}, []
    with h5py.File(WEIGHTS, 'r') as handle:
        config = json.loads(handle.attrs['model_config'])
        for spec in config['config']['layers']:
            name, kind, cfg = spec['name'], spec['class_name'], spec['config']
            parents = [x[0] for x in spec['inbound_nodes'][0]] if spec['inbound_nodes'] else []
            if kind == 'InputLayer':
                assert cfg['batch_input_shape'] == [None, 64, 224, 224, 5]
                tensors[name] = keras.Input(shape=(64, 224, 224, 5), name=name)
                continue
            inputs = [tensors[p] for p in parents]
            if kind == 'Lambda':
                # Explicit safe slices from the published notebook, no bytecode load.
                assert parents == ['input_1'] and name in ('lambda_1', 'lambda_2')
                if name == 'lambda_1':
                    layer = keras.layers.Lambda(lambda x: x[..., :3], name=name)
                else:
                    layer = keras.layers.Lambda(lambda x: x[..., 3:5], name=name)
            elif kind == 'Conv3D':
                keys = ('filters', 'kernel_size', 'strides', 'padding', 'data_format',
                        'dilation_rate', 'activation', 'use_bias')
                layer = keras.layers.Conv3D(name=name, **{k: cfg[k] for k in keys})
            elif kind == 'MaxPooling3D':
                keys = ('pool_size', 'strides', 'padding', 'data_format')
                layer = keras.layers.MaxPooling3D(name=name, **{k: cfg[k] for k in keys})
            elif kind == 'Multiply':
                layer = keras.layers.Multiply(name=name)
            elif kind == 'Flatten':
                layer = keras.layers.Flatten(data_format=cfg['data_format'], name=name)
            elif kind == 'Dense':
                layer = keras.layers.Dense(cfg['units'], activation=cfg['activation'],
                                           use_bias=cfg['use_bias'], name=name)
            elif kind == 'Dropout':
                layer = keras.layers.Dropout(cfg['rate'], name=name)
            else:
                raise ValueError('Unreviewed layer: ' + kind)
            tensors[name] = layer(inputs if kind == 'Multiply' else inputs[0])
            if kind in ('Conv3D', 'Dense'):
                group = handle['model_weights'][name][name]
                values = [np.asarray(group['kernel:0']), np.asarray(group['bias:0'])]
                layer.set_weights(values)
                for actual, original in zip(layer.get_weights(), values, strict=True):
                    np.testing.assert_array_equal(actual, original)
                learned.extend([name + '/kernel:0', name + '/bias:0'])
        expected = []
        handle['model_weights'].visititems(
            lambda n, obj: expected.append(n) if isinstance(obj, h5py.Dataset) else None)
        assert len(expected) == len(learned) == 50
    model = keras.Model(tensors['input_1'], tensors['dense_3'], name='author_fgn')
    assert model.count_params() == 272690, model.count_params()
    model.to(device)
    model.eval()
    return model


def sampling_indices(length, target=64):
    """Exact author ceil-stride sampling and end-padding, including short clips."""
    if length < 1:
        raise ValueError('Empty video window')
    stride = int(math.ceil(length / target))
    result = list(range(0, length, stride))
    for i in range(-(target - len(result)), 0):
        result.append(length + i if -i <= length else 0)
    assert len(result) == target
    return result


def optical_flow(a, b):
    gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).reshape(224, 224, 1)
    gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).reshape(224, 224, 1)
    flow = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, .5, 3, 15, 3, 5, 1.2,
                                      cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    for channel in (0, 1):
        flow[..., channel] -= np.mean(flow[..., channel])
        flow[..., channel] = cv2.normalize(flow[..., channel], None, 0, 255, cv2.NORM_MINMAX)
    # Preserve the author's float -> uint8 .npy serialization boundary.
    return flow.astype(np.uint8)


def normalize(data):
    data = np.asarray(data, dtype=np.float32).copy()
    diagnostics = {}
    for name, channels in [('rgb', slice(0, 3)), ('flow', slice(3, 5))]:
        block = data[..., channels]
        mean, std = float(np.mean(block)), float(np.std(block))
        diagnostics[name] = {'mean': mean, 'std': std}
        if not math.isfinite(std) or std <= 0:
            raise ValueError('Author normalization has zero variance: ' + name)
        data[..., channels] = (block - mean) / std
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite normalized input')
    return data, diagnostics


class VideoInputs:
    """Decode once; calculate the original adjacent-frame flow only where needed.

Shared frames are reused across windows. The last flow of EACH window is zero,
and its final native video frame is omitted as in Video2Npy, not the whole video.
"""
    def __init__(self, row, intervals):
        self.started = time.perf_counter()
        if file_sha(row['path']) != row['sha256']:
            raise ValueError('Source video hash mismatch')
        cap = cv2.VideoCapture(row['path'])
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.specs = []
        wanted, flow_wanted = set(), set()
        try:
            if self.fps <= 0 or self.total < 2:
                raise ValueError('Invalid video metadata')
            for start, end in intervals:
                first = max(0, round(start * self.fps))
                stop = min(self.total, round(end * self.fps))
                length = stop - first - 1  # author's Video2Npy drops final native frame
                ids = [first + i for i in sampling_indices(length)]
                last = stop - 2
                self.specs.append(dict(start=start, end=end, first=first, stop=stop,
                                       last_rgb=last, ids=ids))
                wanted.update(ids)
                flow_wanted.update(i for i in ids if i != last)
            wanted.update(i + 1 for i in flow_wanted)
            self.rgb, self.flow = {}, {}
            decode_start = time.perf_counter()
            for index in range(max(wanted) + 1):
                ok = cap.grab()
                if not ok:
                    raise ValueError(f'Video decoding failed at frame {index}')
                if index not in wanted:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError(f'Video retrieval failed at frame {index}')
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                self.rgb[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.decode_seconds = time.perf_counter() - decode_start
            flow_start = time.perf_counter()
            for index in sorted(flow_wanted):
                self.flow[index] = optical_flow(self.rgb[index], self.rgb[index + 1])
            self.flow_seconds = time.perf_counter() - flow_start
            self.preparation_seconds = time.perf_counter() - self.started
        finally:
            cap.release()

    def window(self, index):
        spec = self.specs[index]
        out = np.zeros((64, 224, 224, 5), dtype=np.uint8)
        for j, source in enumerate(spec['ids']):
            out[j, ..., :3] = self.rgb[source]
            if source != spec['last_rgb']:
                out[j, ..., 3:] = self.flow[source]
        result, diagnostics = normalize(out)
        return result, {'native_frame_indices': spec['ids'], 'normalization': diagnostics,
                        'input_sha256': hashlib.sha256(result.tobytes()).hexdigest()}


def score(model, data, device='cuda'):
    import keras
    import torch
    value = torch.from_numpy(data[None]).to(device)
    with torch.inference_mode(), keras.device(device):
        output = model(value, training=False)
    probabilities = output.detach().cpu().numpy()[0]
    if not np.isfinite(probabilities).all() or abs(float(probabilities.sum()) - 1) > 1e-5:
        raise ValueError('Invalid model output')
    # sorted(['Fight', 'NonFight']) in the author training generator.
    return float(probabilities[0]), probabilities.tolist()
