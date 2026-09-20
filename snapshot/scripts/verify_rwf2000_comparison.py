"""Independent checks against the reviewed author's preprocessing notebook."""
from pathlib import Path
import ast
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.rwf2000_model import SOURCE, WEIGHTS, VideoInputs, load_model, score, sampling_indices, normalize, file_sha
from scripts.evaluate_rwf2000_comparison import OUT, MANIFEST, init_runtime, write, read, aggregate
from scripts.skeleton_round2 import choose_threshold
import cv2
import numpy as np
import torch


def author_reference():
    """Compile only the two reviewed definitions, excluding imports/training code."""
    env = {'np': np, 'cv2': cv2, 'os': __import__('os')}
    prep = read(SOURCE / 'Preprocess/Video2Numpy.ipynb')
    code = '\n'.join(''.join(c['source']) for c in prep['cells'] if c['cell_type'] == 'code')
    tree = ast.parse(code)
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'getOpticalFlow']
    assert len(defs) == 1
    exec(compile(ast.Module(body=defs, type_ignores=[]), '<reviewed author optical flow>', 'exec'), env)
    network = read(SOURCE / 'Networks/Flow Gated Network.ipynb')
    code = '\n'.join(''.join(c['source']) for c in network['cells'] if c['cell_type'] == 'code')
    tree = ast.parse(code)
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DataGenerator')
    original.bases = []
    original.body = [n for n in original.body if isinstance(n, ast.FunctionDef)
                     and n.name in ('normalize', 'uniform_sampling', 'load_data')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[])),
                 '<reviewed author data loader>', 'exec'), env)
    generator = env['DataGenerator']()
    generator.data_aug = False
    return env['getOpticalFlow'], generator


def main():
    init_runtime()
    flow, generator = author_reference()
    sample_cases = [1, 2, 5, 31, 63, 64, 65, 100, 119, 120, 149, 150, 151, 300]
    for n in sample_cases:
        video = np.arange(n, dtype=np.float32)
        np.testing.assert_array_equal(generator.uniform_sampling(video), np.array(sampling_indices(n)))
    # Same-protocol normalization failure must not silently become a normal label.
    try:
        normalize(np.zeros((64, 2, 2, 5), dtype=np.uint8))
    except ValueError:
        pass
    else:
        raise AssertionError('Zero variance not detected')
    assert aggregate([dict(eligible=True, score=None), dict(eligible=True, score=.1)], 'score') is None
    assert aggregate([dict(eligible=False, score=.8)], 'score', True) is None
    th, met = choose_threshold([0, 1], [1., 1.])
    assert th > 1 and met['recall'][1] == 0
    rows = [r for r in read(MANIFEST)['rows'] if r['split'] == 'validation']
    # Select by stable identifier, not by labels, scores or apparent success.
    row = min(rows, key=lambda r: r['sample_id'])
    interval = (0., min(row['duration'], 4.))
    actual = VideoInputs(row, [interval])
    spec = actual.specs[0]
    cap = cv2.VideoCapture(row['path'])
    frames = []
    for index in range(spec['last_rgb'] + 1):
        ok, frame = cap.read()
        assert ok
        if index >= spec['first']:
            frames.append(cv2.cvtColor(cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    cap.release()
    frames = np.array(frames)
    reference_flow = flow(frames)
    original = np.zeros((*frames.shape[:-1], 5))
    original[..., :3], original[..., 3:] = frames, reference_flow
    original = original.astype(np.uint8)
    expected = generator.uniform_sampling(np.float32(original), target_frames=64)
    expected[..., :3] = generator.normalize(expected[..., :3])
    expected[..., 3:] = generator.normalize(expected[..., 3:])
    data, audit = actual.window(0)
    np.testing.assert_array_equal(data, expected)
    del original, expected, frames, reference_flow, actual
    print('Author preprocessing exact match; sampling edge cases passed', flush=True)
    model = load_model('cuda')
    t = time.perf_counter()
    gpu, gpu_probs = score(model, data, 'cuda')
    gpu2, _ = score(model, data, 'cuda')
    assert gpu == gpu2
    model.to('cpu')
    cpu, cpu_probs = score(model, data, 'cpu')
    np.testing.assert_allclose(cpu_probs, gpu_probs, atol=1e-4, rtol=1e-4)
    print('CPU/GPU parity passed', cpu_probs, gpu_probs, 'seconds', time.perf_counter() - t, flush=True)
    write(OUT / 'verification.json', dict(
        passed=True, model_sha256=file_sha(WEIGHTS), learned_tensors=50,
        parameter_count=model.count_params(), author_preprocessing_exact=True,
        sampler_lengths=sample_cases, zero_variance_handling=True, missing_window_handling=True,
        threshold_above_one_endpoint=True, gpu_repeat_exact=True,
        cpu_probabilities=cpu_probs, gpu_probabilities=gpu_probs,
        cpu_gpu_max_abs_difference=float(np.max(np.abs(np.array(cpu_probs) - gpu_probs))),
        reference_sample_id=row['sample_id'], reference_input_sha256=audit['input_sha256'],
        runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                     opencv=cv2.__version__, keras=__import__('keras').__version__, device=torch.cuda.get_device_name(0)),
        source_verification_script_sha256=file_sha(__file__)))


if __name__ == '__main__':
    main()
