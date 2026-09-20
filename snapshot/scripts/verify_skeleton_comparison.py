"""Verify provenance, train/test isolation and inference parity using only local artifacts."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import torch
import cv2
from scripts.skeleton_common import *
from scripts.skeleton_io import write
from scripts.train_skeleton_comparison import make_rgb
from scripts.evaluate_skeleton_comparison import evaluate_sample,summarize
from backend.vision.skeleton_actions import SkeletonActionModel


def verify(external=False):
    offline()
    torch.set_num_threads(4)
    report = {'status': 'running', 'checks': [], 'inference_parity': {}}
    manifests = {n: read(CACHE / 'manifests' / f'{n}.json') for n in ['gmd', 'tnue', 'fallvision', 'vfd']}
    for name, manifest in manifests.items():
        isolation(manifest['rows'])
        report['checks'].append({'dataset': name, 'manifest_sha256': sha(CACHE / 'manifests' / f'{name}.json'),
                                 'included': len(manifest['rows']), 'split_isolation': True})
    for task, ext in [('gmd', 'fallvision'), ('tnue', 'vfd')]:
        hashes = {r['sha256'] for r in manifests[task]['rows']}
        assert not hashes.intersection(r['sha256'] for r in manifests[ext]['rows']), 'Cross-source hash overlap'
        models, thresholds, stored = {}, {}, {}
        for modality in ['rgb', 'skeleton']:
            for seed in [42, 43, 44]:
                base = OUT / task / modality / f'seed_{seed}'
                selection = read(base / 'selection.json')
                config = read(base / 'config.json')
                training_record=read(base/'run_record.json')
                epochs_run=sum(len(stage['history']) for stage in training_record['stages'].values())
                assert epochs_run<=50, 'Actual training exceeded the requested total epoch budget'
                assert selection['sha256'] == sha(base / 'selected_best.pt')
                assert config['manifest_sha256'] == sha(CACHE / 'manifests' / f'{task}.json')
                assert config['pose_sha256'] == sha(POSE)
                assert config['protocol'] == PROTOCOL and config['seed'] == seed
                for file, expected in config['code_hashes'].items():
                    assert sha(ROOT / file) == expected, f'Training code changed: {file}'
                saved = torch.load(base / 'selected_best.pt', map_location='cpu', weights_only=True)
                assert selection['threshold'] == saved['threshold']
                model = make_rgb(False) if modality == 'rgb' else SkeletonActionModel('fall' if task == 'gmd' else 'fight')
                model.load_state_dict(saved['state_dict'])
                key = f'{modality}_{seed}'
                models[key] = model.cuda().eval()
                thresholds[key] = saved['threshold']
                predictions = read(base / 'test_predictions.json')['predictions']
                stored[key] = {p['sample_id']: p for p in predictions}
                expected_ids = {r['sample_id'] for r in manifests[task]['rows'] if r['split'] == 'test'}
                assert len(predictions) == len(expected_ids) and set(stored[key]) == expected_ids
                report['checks'].append({'model':f'{task}/{modality}/{seed}',
                    'actual_epochs_all_stages':epochs_run,'selected_checkpoint_sha256':selection['sha256'],
                    'threshold_matches_saved_validation_selection':True})
        differences, changed, checked, detailed = [], [], 0, []
        for row in manifests[task]['rows']:
            if row['split'] != 'test':
                continue
            cache = torch.load(CACHE / task / (row['sample_id'] + '.pt'), map_location='cpu', weights_only=True)
            assert cache['source_sha256'] == row['sha256'] and cache['protocol'] == PROTOCOL
            result = evaluate_sample(row, cache, models, thresholds)
            detailed.append(result)
            for key, actual in result['results'].items():
                original = stored[key][row['sample_id']]
                assert original['label'] == row['label']
                assert (actual['score'] is None) == (original['score'] is None)
                if actual['score'] is not None:
                    differences.append(abs(actual['score'] - original['score']))
                if actual['prediction'] != original['prediction']:
                    changed.append({'sample_id': row['sample_id'], 'model': key, 'original': original, 'actual': actual})
                checked += 1
        report['inference_parity'][task] = {'predictions_checked': checked,
            'max_absolute_score_difference': max(differences or [0]), 'changed_predictions': changed}
        write(OUT/'evaluations'/f'internal_{task}.json',dict(records=detailed,
            summary=summarize(detailed,list(models),len(detailed)),
            interpretation='Internal test only; all six model predictions checked against stored training outputs.'))
        print(task, report['inference_parity'][task], flush=True)
        del models, model
        torch.cuda.empty_cache()
        if external:
            summary = read(OUT / 'evaluations' / task / 'summary.json')
            expected = {r['sample_id'] for r in manifests[ext]['rows']}
            paths = list((OUT / 'evaluations' / task / 'samples').glob('*.json'))
            assert len(paths) == len(expected) and {p.stem for p in paths} == expected
            assert summary['samples'] == len(expected)
            for key, metrics_value in summary['per_seed'].items():
                assert np.array(metrics_value['confusion_matrix']).sum() == len(expected)
            report['checks'].append({'external_dataset': ext, 'all_six_models_same_samples': True,
                                     'samples': len(expected)})
    if external:
        videos=read(OUT/'acceptance/videos.json')
        decoded=0
        for media in videos:
            assert Path(media['source']).is_file() and Path(media['poster']).is_file()
            cap=cv2.VideoCapture(media['video']);frames=0
            while True:
                ok,frame=cap.read()
                if not ok:break
                assert frame.shape[:2]==(416,1280)
                frames+=1
            cap.release()
            assert frames==32, f'Incomplete acceptance video: {media["video"]}'
            decoded+=frames
        report['checks'].append({'acceptance_videos':len(videos),'decoded_frames':decoded,
                                 'all_original_video_links_exist':True})
    report['artifact_code_hashes']={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'scripts').glob('*skeleton*.py')}
    report['status'] = 'passed' if all(not r['changed_predictions'] and r['max_absolute_score_difference'] < .005
                                      for r in report['inference_parity'].values()) else 'requires_review'
    write(OUT / ('final_verification.json' if external else 'internal_verification.json'), report)
    assert report['status'] == 'passed', 'Prediction parity requires review'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--external', action='store_true')
    print(verify(parser.parse_args().external), flush=True)
