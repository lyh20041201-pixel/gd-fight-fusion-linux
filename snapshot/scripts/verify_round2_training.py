"""Synthetic dropout/gradient/resume checks; RGB parity uses training rows only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copy
import torch,numpy as np
from scripts.skeleton_common import offline,read,write,rgb_tensor,sha
from scripts.skeleton_round2 import OUT,MANIFESTS
from scripts.train_skeleton_round2 import rng_state,restore_rng,training_logit,window_logit,SampleStore,make_rgb,checkpoint


def synthetic_data():
    values=[]
    for shift in [0.,.3,-.2]:
        values.append(dict(features=torch.randn(2,6,32,17)+shift,valid=torch.ones(2,32,dtype=torch.bool),
            centers=torch.randn(2,32,2)*30,scale=torch.ones(2)*100,times=torch.linspace(0,4,32)))
    return values


def main():
    offline();torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    from backend.vision.skeleton_actions import SkeletonActionModel
    checks=[]
    for task in ['fall','fight']:
        torch.manual_seed(918);torch.cuda.manual_seed_all(918)
        data=synthetic_data();a=SkeletonActionModel(task).cuda().train();b=copy.deepcopy(a)
        before=rng_state()
        with torch.autocast('cuda'):
            direct=torch.stack([window_logit(a,data,'skeleton','baseline',i) for i in range(3)]).max()
            loss=torch.nn.functional.binary_cross_entropy_with_logits(direct,torch.ones((),device='cuda'))
        loss.backward();after=rng_state()
        restore_rng(before)
        with torch.autocast('cuda'):
            replay=training_logit(b,data,'skeleton','baseline')
            other=torch.nn.functional.binary_cross_entropy_with_logits(replay,torch.ones((),device='cuda'))
        other.backward();replayed_after=rng_state()
        assert float(direct)==float(replay)
        diffs=[]
        for x,y in zip(a.parameters(),b.parameters()):
            if x.grad is None:assert y.grad is None;continue
            torch.testing.assert_close(x.grad,y.grad,atol=1e-6,rtol=1e-5)
            diffs.append(float((x.grad-y.grad).abs().max()))
        assert torch.equal(after['cuda'][0],replayed_after['cuda'][0])
        checks.append(dict(check='bounded_MIL_equals_full_graph',task=task,max_gradient_difference=max(diffs),same_dropout_rng=True))
    # Actual optimizer/scaler/RNG serialization, then the same next stochastic step.
    model=SkeletonActionModel('fight').cuda().train();opt=torch.optim.AdamW(model.parameters(),lr=.001);scaler=torch.amp.GradScaler('cuda')
    data=synthetic_data()
    def step(m,o,s):
        o.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            v=training_logit(m,data,'skeleton','baseline');loss=torch.nn.functional.binary_cross_entropy_with_logits(v,torch.ones((),device='cuda'))
        s.scale(loss).backward();s.unscale_(o);torch.nn.utils.clip_grad_norm_(m.parameters(),5.)
        s.step(o);s.update();return float(loss)
    step(model,opt,scaler)
    path=OUT/'verification/synthetic_resume.pt';checkpoint(path,model,opt,scaler,{'cursor':4,'stage_index':0,'epoch':0},'synthetic-v1')
    uninterrupted=step(model,opt,scaler);expected={k:v.clone() for k,v in model.state_dict().items()}
    saved=torch.load(path,map_location='cpu',weights_only=True)
    restored=SkeletonActionModel('fight').cuda().train();restored.load_state_dict(saved['model'])
    op2=torch.optim.AdamW(restored.parameters(),lr=.001);op2.load_state_dict(saved['optimizer'])
    sc2=torch.amp.GradScaler('cuda');sc2.load_state_dict(saved['scaler']);restore_rng(saved['rng'])
    resumed=step(restored,op2,sc2)
    assert uninterrupted==resumed
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in expected.items())
    checks.append(dict(check='optimizer_scaler_rng_resume_next_step',loss_exact=True,all_model_tensors_exact=True,source='synthetic; no real heldout test sample'))
    # Bounded feature-cache and dense RGB paths, without touching retained test.
    manifest=read(MANIFESTS/'vfd.json');selected=[]
    for label in [0,1]:selected.append(next(r for r in manifest['rows'] if r['split']=='train' and r['label']==label))
    store=SampleStore('vfd',selected,'rgb',allow_feature_build=True);model=make_rgb().cuda().eval();maximum=0.
    for row in selected:
        features=store.get(row);raw=store.raw(row)
        for i,c in enumerate(raw['clips'][:2]):
            with torch.inference_mode(),torch.autocast('cuda'):
                v=model(rgb_tensor(c['rgb']).unsqueeze(0).cuda());direct=v[0,1]-v[0,0]
                for stage in ['baseline','finetune']:
                    cached=window_logit(model,features,'rgb',stage,i)
                    diff=abs(float(direct)-float(cached));maximum=max(maximum,diff)
                    assert diff<.002
    store.close_extractor()
    checks.append(dict(check='dense_vs_cached_RGB',training_sample_ids=[r['sample_id'] for r in selected],max_logit_difference=maximum,test_samples_used=0))
    result=dict(status='passed',checks=checks,code_sha256=sha(__file__),test_set_accessed=False)
    write(OUT/'verification/training_checks.json',result);print(result,flush=True)


if __name__=='__main__':main()
