from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import backend.vision.fight_fusion as fusion


def make_clip(people=2, frames=8, confidence=.2):
    keypoints = torch.zeros(people, frames, 17, 3)
    boxes = torch.zeros(people, frames, 4)
    for person in range(people):
        boxes[person] = torch.tensor([10 + 25 * person, 20, 30 + 25 * person, 70])
        keypoints[person, ..., 0] = 20 + 25 * person
        keypoints[person, ..., 1] = 45
        keypoints[person, ..., 2] = confidence
    return dict(keypoints=keypoints, boxes=boxes, frame_indices=list(range(frames)),
                timestamps=[i * .125 for i in range(frames)], track_ids=list(range(people)))


def test_soft_pose_preserves_confidence_and_only_discards_very_weak_xy():
    clip = make_clip()
    clip['keypoints'][:, :, 0, 2] = .005
    item = fusion.prepare_fusion_skeleton(clip, (100, 100))
    assert item['features'].shape == (2, 3, 8, 17)
    assert torch.all(item['features'][:, :2, :, 0] == 0)
    assert torch.all(item['features'][:, 2, :, 0] == .005)
    assert torch.all(item['features'][:, 2, :, 1] == .2)
    assert item['valid'].all()


def test_short_and_repeated_frames_do_not_create_skeleton_evidence():
    assert len(fusion.prepare_fusion_skeleton(make_clip(frames=3), (100, 100))['features']) == 0
    clip = make_clip()
    clip['frame_indices'] = [0, 0, 0, 1, 1, 2, 2, 2]
    clip['timestamps'] = [i * .125 for i in clip['frame_indices']]
    assert len(fusion.prepare_fusion_skeleton(clip, (100, 100))['features']) == 0
    assert fusion.interaction_rois(clip, (100, 100)) == []
    assert fusion.window_quality(clip, (100, 100), [])['unique_frame_fraction'] == 3 / 8
    clip['timestamps'] = [0.] * 8
    with pytest.raises(ValueError, match='same repetitions'):
        fusion.prepare_fusion_skeleton(clip, (100, 100))


def test_pair_requires_four_shared_unique_frames_not_just_four_per_person():
    clip = make_clip(frames=8)
    clip['keypoints'][0, 4:, :, 2] = 0
    clip['keypoints'][1, :4, :, 2] = 0
    item = fusion.prepare_fusion_skeleton(clip, (100, 100))
    model = fusion.FusionSkeletonModel().eval()
    assert len(item['features']) == 2
    assert model(item) is None


def test_roi_uses_boxes_even_when_pose_missing_and_clamps_padding():
    clip = make_clip(confidence=0)
    rois = fusion.interaction_rois(clip, (80, 60))
    assert len(rois) == 1
    assert rois[0]['box'] == [1, 10, 60, 80]
    images = [np.full((80, 60, 3), [10, 20, 30], np.uint8) for _ in range(8)]
    crops = fusion.crop_rgb_frames(images, rois)
    assert crops[0].shape == (8, 112, 112, 3)
    assert np.array_equal(crops[0][0, 56, 56], [30, 20, 10])
    assert fusion.interaction_rois(clip, (80, 60), max_rois=0) == []
    with pytest.raises(ValueError, match='outside'):
        fusion.crop_rgb_frames(images, [dict(box=[-1, 0, 10, 10])])


class SymmetricBackbone(nn.Module):
    def forward(self, x):
        # Test the real pair head/geometry without spending CPU on graph blocks.
        # Input N,M,T,V,C; output N,M,C,T,V.
        return x[..., :1].permute(0, 1, 4, 2, 3).repeat(1, 1, 256, 1, 1)


def test_skeleton_pair_score_and_roi_selection_are_track_order_symmetric():
    torch.manual_seed(4)
    clip = make_clip(people=3)
    item = fusion.prepare_fusion_skeleton(clip, (100, 100))
    model = fusion.FusionSkeletonModel().eval()
    model.backbone = SymmetricBackbone()
    with torch.inference_mode():
        first = model(item)
        reversed_clip = dict(clip, keypoints=clip['keypoints'].flip(0), boxes=clip['boxes'].flip(0),
                             track_ids=list(reversed(clip['track_ids'])))
        second = model(fusion.prepare_fusion_skeleton(reversed_clip, (100, 100)))
    assert torch.allclose(first, second, atol=1e-6)
    assert [r['box'] for r in fusion.interaction_rois(clip, (100, 100))] == [
        r['box'] for r in fusion.interaction_rois(reversed_clip, (100, 100))]


def test_missing_features_neutral_values_masks_interactions_and_dropout():
    quality = fusion.window_quality(make_clip(), (100, 100), [])
    value = fusion.fusion_features([2., None, float('nan')], quality, [True, False, False])
    assert len(value) == len(fusion.FUSION_FEATURE_NAMES) == 38
    assert value[:6].tolist() == [2, 0, 0, 1, 0, 0]
    assert not value[22:].any()
    with pytest.raises(ValueError, match='finite logit'):
        fusion.fusion_features([2, None, 1], quality, [True, True, True])
    present = fusion.fusion_features([2., 3., 4.], quality, [True, True, True])
    model = fusion.QualityFusion(missing_dropout=1).train()
    observed = []
    handle = model.linear.register_forward_pre_hook(lambda _, args: observed.append(args[0].detach()))
    model(present)
    handle.remove()
    assert torch.equal(observed[0], value)
    model.set_normalization(torch.stack([value, present]))
    assert torch.isfinite(model.feature_std).all() and (model.feature_std > 0).all()
    model.set_normalization(torch.stack([present, present]))
    assert torch.equal(model.feature_std, torch.ones_like(model.feature_std))


def test_no_pose_still_runs_full_rgb_and_fusion(monkeypatch):
    calls = []
    predictor = object.__new__(fusion.FightFusionPredictor)
    predictor.config = {}
    predictor.device = torch.device('cpu')
    predictor.threshold = .5
    predictor.global_model, predictor.roi_model = object(), object()
    predictor.skeleton_model = lambda item: None
    predictor.fusion_model = fusion.QualityFusion().eval()
    with torch.no_grad():
        predictor.fusion_model.linear.weight.zero_()
        predictor.fusion_model.linear.weight[0, 0] = 1
        predictor.fusion_model.linear.bias.zero_()
    def score(model, rgb):
        calls.append(model)
        return torch.tensor(2.)
    monkeypatch.setattr(fusion, 'rgb_logit', score)
    clip = make_clip(people=0)
    images = [np.zeros((100, 100, 3), np.uint8) for _ in range(8)]
    result = predictor.score_window(images, clip, (100, 100))
    assert calls == [predictor.global_model]
    assert result['availability'] == {'global': True, 'roi': False, 'skeleton': False}
    assert result['branch_logits'] == {'global': 2., 'roi': None, 'skeleton': None}
    assert result['score'] == pytest.approx(torch.sigmoid(torch.tensor(2.)).item())


class FakePose:
    device = torch.device('cpu')
    def __init__(self):
        self.calls = []
        self.frames = 0
    def predict(self, images, **kwargs):
        self.calls.append(kwargs)
        results = []
        for _ in images:
            count = 0 if self.frames == 2 else 1
            # Some YOLO releases preserve flattened width on an empty result.
            points = torch.ones(count, 17, 3) if count else torch.empty(0, 51)
            score = .8 if self.frames == 0 else .15
            boxes = torch.tensor([[10., 20., 30., 70.]])[:count]
            results.append(SimpleNamespace(keypoints=SimpleNamespace(data=points),
                                           boxes=SimpleNamespace(xyxy=boxes, conf=torch.full((count,), score))))
            self.frames += 1
        return results


def test_shared_extractor_uses_low_confidence_recovery_and_only_actual_observations():
    pose = FakePose()
    images = [np.zeros((100, 100, 3), np.uint8) for _ in range(5)]
    result = fusion.extract_pose_clip(images, [0., .125, .25, .375, .375], [0, 1, 2, 3, 3], pose)
    assert pose.frames == 4
    assert pose.calls[0]['conf'] == .1 and pose.calls[0]['half'] is False
    assert result['boxes'].shape == (1, 5, 4)
    assert result['boxes'][0, 1].any()  # .15 observation recovered the .8 track
    assert not result['boxes'][0, 2].any()  # no predicted box substituted
    assert torch.equal(result['boxes'][0, 3], result['boxes'][0, 4])
    with pytest.raises(ValueError, match='original timestamp'):
        fusion.extract_pose_clip(images[:2], [0., .1], [0, 0], pose)


def test_raw_pose_adapter_preserves_weak_xy_but_standard_results_stay_masked(monkeypatch):
    from ultralytics.utils import ops

    predictor_type = fusion.raw_pose_predictor_type()
    predictor = object.__new__(predictor_type)
    predictor.args = SimpleNamespace(conf=.1, iou=.45, agnostic_nms=False, max_det=100, classes=None)
    predictor.model = SimpleNamespace(names={0: 'person'}, kpt_shape=(17, 3))
    predictor.batch = (['frame.jpg'],)
    points = torch.tensor([20., 45., .2])[None].repeat(17, 1)
    points[:5, 2] = torch.tensor([.005, .1, .49, .5, .9])
    prediction = torch.cat([torch.tensor([10., 20., 30., 70., .8, 0.]), points.flatten()])[None]
    calls = []

    def nms(preds, conf, iou, **kwargs):
        calls.append((conf, iou, kwargs))
        return [prediction.clone()]

    monkeypatch.setattr(ops, 'non_max_suppression', nms)
    result = predictor.postprocess(None, torch.zeros(1, 3, 100, 100),
                                   [np.zeros((100, 100, 3), np.uint8)])[0]
    torch.testing.assert_close(result.fusion_raw_keypoints[0], points)
    torch.testing.assert_close(result.keypoints.data[0, :, 2], points[:, 2])
    assert not result.keypoints.data[0, :3, :2].any()
    torch.testing.assert_close(result.keypoints.data[0, 3:5, :2], points[3:5, :2])
    assert result.fusion_raw_keypoints.data_ptr() != result.keypoints.data.data_ptr()
    assert calls == [(.1, .45, dict(agnostic=False, max_det=100, classes=None, nc=1))]


def test_raw_pose_adapter_empty_observations_have_explicit_coco_shape(monkeypatch):
    from ultralytics.utils import ops

    predictor_type = fusion.raw_pose_predictor_type()
    predictor = object.__new__(predictor_type)
    predictor.args = SimpleNamespace(conf=.1, iou=.45, agnostic_nms=False, max_det=100, classes=None)
    predictor.model = SimpleNamespace(names={0: 'person'}, kpt_shape=(17, 3))
    predictor.batch = (['frame.jpg'],)
    monkeypatch.setattr(ops, 'non_max_suppression', lambda *args, **kwargs: [torch.empty(0, 57)])
    result = predictor.postprocess(None, torch.zeros(1, 3, 100, 100),
                                   [np.zeros((100, 100, 3), np.uint8)])[0]
    assert result.fusion_raw_keypoints.shape == (0, 17, 3)
    assert result.keypoints.data.shape == (0, 17, 3)
    assert result.boxes.xyxy.shape == (0, 4)


def test_candidate_extract_reads_raw_extra_without_affecting_fake_pose_support():
    class RawPose(FakePose):
        def predict(self, images, **kwargs):
            results = super().predict(images, **kwargs)
            for result in results:
                count = len(result.boxes.xyxy)
                raw = torch.tensor([20., 45., .2])[None, None].repeat(count, 17, 1)
                result.fusion_raw_keypoints = raw
                result.keypoints.data = raw.clone()
                result.keypoints.data[..., :2] = 0
            return results

    pose = RawPose()
    images = [np.zeros((100, 100, 3), np.uint8) for _ in range(6)]
    clip = fusion.extract_pose_clip(images, [i / 8 for i in range(6)], list(range(6)), pose)
    observed = clip['boxes'][..., 2] > clip['boxes'][..., 0]
    assert (clip['keypoints'][observed][..., 0] == 20).all()
    assert (clip['keypoints'][observed][..., 1] == 45).all()
    assert (clip['keypoints'][observed][..., 2] == .2).all()
    prepared = fusion.prepare_fusion_skeleton(clip, (100, 100))
    assert prepared['valid'].sum() == 5
    torch.testing.assert_close(prepared['features'][0, :2, 0, 7], torch.tensor([-.6, -.1]))


def test_adapter_replaces_already_initialized_yolo_predictor_without_global_patch(monkeypatch):
    from ultralytics import YOLO
    from ultralytics.models.yolo.pose.predict import PosePredictor

    original_postprocess = PosePredictor.postprocess
    yolo = object.__new__(YOLO)
    nn.Module.__init__(yolo)
    yolo.model = nn.Linear(1, 1)
    yolo.overrides = dict(task='pose')
    yolo.callbacks = {}
    old_predictor = SimpleNamespace(args=SimpleNamespace(conf=.25))
    yolo.predictor = old_predictor
    calls = []

    class CandidatePredictor:
        def __init__(self, overrides, _callbacks):
            self.args = SimpleNamespace(**overrides)
            calls.append(('construct', overrides, _callbacks))

        def setup_model(self, model, verbose):
            self.model = model
            calls.append(('setup', model, verbose))

    monkeypatch.setattr(fusion, 'raw_pose_predictor_type', lambda: CandidatePredictor)
    assert fusion.ensure_raw_pose_predictor(yolo)
    assert isinstance(yolo.predictor, CandidatePredictor)
    assert yolo.predictor is not old_predictor
    installed = yolo.predictor
    assert fusion.ensure_raw_pose_predictor(yolo)
    assert yolo.predictor is installed and len(calls) == 2
    assert calls[0][1]['save'] is False
    assert calls[1] == ('setup', yolo.model, False)
    assert PosePredictor.postprocess is original_postprocess
    assert not fusion.ensure_raw_pose_predictor(FakePose())


def test_checkpoint_hash_is_checked_before_deserialization(tmp_path):
    path = tmp_path / 'bad.pt'
    path.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError, match='hash mismatch'):
        fusion._load_verified(dict(path=str(path), sha256='0' * 64), nn.Linear(1, 1), 'cpu')
