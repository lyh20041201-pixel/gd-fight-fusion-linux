from types import SimpleNamespace

import numpy as np
import pytest

from backend.vision import fight_fusion_live as live


def frames(times):
    return [(float(t),np.zeros((64,96,3),dtype=np.uint8)) for t in times]


class Predictor:
    threshold=.75
    device='cpu'
    config={'label':'test candidate'}

    def __init__(self):
        self.calls=0

    def score_window(self, images, clip, shape):
        self.calls+=1
        assert len(images)==len(clip['timestamps'])==32
        assert not clip['track_ids']
        return dict(score=.8,is_fight=True,branch_logits={'global':2.,'roi':None,'skeleton':None},
                    availability={'global':True,'roi':False,'skeleton':False},quality={},rois=[])


def adapter(monkeypatch):
    predictor=Predictor()
    monkeypatch.setattr(live,'assess_view',lambda xs:dict(ok=True))
    monkeypatch.setattr(live,'extract_pose_clip',lambda images,timestamps,indices,pose,camera_id:
                        dict(timestamps=timestamps,frame_indices=indices,track_ids=[]))
    return live.FightFusionLiveAdapter({},predictor=predictor,pose_model=object()),predictor


def test_live_pose_missing_still_scores_global(monkeypatch):
    model,predictor=adapter(monkeypatch)
    result=model.predict('camera',frames(np.linspace(100,104,33)))
    assert predictor.calls==1
    assert result['actions'][0]['state']=='candidate'
    assert result['actions'][0]['event_type']=='suspected_fight'
    assert result['actions'][0]['availability']['skeleton'] is False
    assert result['actions'][0]['start']>=100
    assert result['actions'][0]['end']==104


@pytest.mark.parametrize('times',[[0,.2,.4], [0,1,2,3,4], list(np.linspace(0,1,16))+list(np.linspace(2,4,16))])
def test_live_incomplete_or_broken_window_does_not_score(monkeypatch,times):
    model,predictor=adapter(monkeypatch)
    assert model.predict('camera',frames(times))['state']=='warming'
    assert predictor.calls==0


def test_live_shape_change_and_view_failure_are_explicit(monkeypatch):
    model,predictor=adapter(monkeypatch)
    values=frames(np.linspace(0,4,33))
    values[5]=(values[5][0],np.zeros((100,200,3),dtype=np.uint8))
    assert model.predict('camera',values)['state']=='warming'
    monkeypatch.setattr(live,'assess_view',lambda xs:dict(ok=False,reason='dark'))
    result=model.predict('camera',frames(np.linspace(0,4,33)))
    assert result['state']=='blocked' and result['reason']=='dark'
    assert predictor.calls==0
