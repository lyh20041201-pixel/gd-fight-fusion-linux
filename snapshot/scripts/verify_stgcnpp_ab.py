"""Synthetic forward/gradient/BN/RNG/resume checks; never uses heldout samples."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ast,hashlib,copy
from scripts.stgcnpp_ab import *

def synthetic(count=3,same=False):
    items=[]
    for i in range(count):
        item=dict(features=torch.randn(2,3,32,17)*.2,valid=torch.ones(2,32,dtype=torch.bool),
            centers=torch.randn(2,32,2)*20,scale=torch.ones(2)*100,times=torch.linspace(0,3.9,32))
        items.append(copy.deepcopy(items[0]) if same and items else item)
    return dict(items=items,usable=True)

def check_source():
    notice=read(ROOT/'third_party/STGCNPP_NOTICE.json');source=(ROOT/'backend/vision/stgcnpp_backbone.py').read_text(encoding='utf-8')
    tree=ast.parse(source);nodes={n.name:n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    for row in notice['verbatim_bodies']:
        body=ast.get_source_segment(source,nodes[row['symbol']])
        assert hashlib.sha256(body.encode()).hexdigest()==row['body_sha256']
    assert sha(ROOT/'backend/vision/stgcnpp_backbone.py')==notice['file_sha256']
    return dict(check='verbatim_upstream_compute_bodies',symbols=len(notice['verbatim_bodies']),commit=notice['commit'])

def main():
    setup_runtime();checks=[check_source()]
    for task in ['fall','fight']:
        for seed in [42,43,44]:
            a,ia=make_model(task,'A',seed);b,ib=make_model(task,'B',seed)
            assert ia['head_initial_sha256']==ib['head_initial_sha256'] and ia['random_initial_model_sha256']==ib['random_initial_model_sha256']
            assert ib['load']['backbone_tensors_loaded']==690
            checks.append(dict(check='paired_initialization',task=task,seed=seed,head_sha256=ia['head_initial_sha256'],loaded_backbone_tensors=690))
    for task,arm,same in [('fall','A',False),('fall','B',True),('fight','A',False),('fight','B',False)]:
        a,_=make_model(task,arm,991);a=a.cuda().train();b=copy.deepcopy(a);data=synthetic(same=same);before=rng_state()
        with torch.autocast('cuda'):
            direct=torch.stack([window_logit(a,item).float() for item in data['items']]).max()
            loss=torch.nn.functional.binary_cross_entropy_with_logits(direct,torch.ones((),device='cuda'))
        loss.backward();after=rng_state();restore_rng(before)
        with torch.autocast('cuda'):
            replay=training_logit(b,data)
            other=torch.nn.functional.binary_cross_entropy_with_logits(replay,torch.ones((),device='cuda'))
        other.backward();replayed=rng_state()
        assert float(direct)==float(replay),(task,arm,float(direct),float(replay))
        maxdiff=0.
        for name,x in a.named_parameters():
            y=dict(b.named_parameters())[name]
            if x.grad is None:assert y.grad is None;continue
            assert y.grad is not None,('missing_replay_gradient',task,arm,name)
            torch.testing.assert_close(x.grad,y.grad,atol=2e-5,rtol=2e-3)
            maxdiff=max(maxdiff,float((x.grad-y.grad).abs().max()))
        assert all(torch.equal(x,b.state_dict()[name]) for name,x in a.state_dict().items())
        assert torch.equal(after['torch'],replayed['torch']) and all(torch.equal(x,y) for x,y in zip(after['cuda'],replayed['cuda']))
        checks.append(dict(check='bounded_MIL_full_graph',task=task,arm=arm,tied_windows=same,max_gradient_difference=maxdiff,
            logits_exact=True,bn_state_exact=True,rng_exact=True))
        del a,b;torch.cuda.empty_cache()
    # Actual serialization followed by an identical next optimizer update.
    model,_=make_model('fight','B',122);model=model.cuda().train();data=synthetic(count=2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);scaler=torch.amp.GradScaler('cuda')
    def step(m,o,s):
        m.train();o.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            logit=training_logit(m,data);loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,torch.ones((),device='cuda'))
        s.scale(loss).backward();s.unscale_(o);torch.nn.utils.clip_grad_norm_(m.parameters(),5.);s.step(o);s.update()
        return float(loss)
    step(model,optimizer,scaler)
    path=OUT/'verification/synthetic_resume.pt';checkpoint(path,model,optimizer,scaler,dict(cursor=4,epoch=0),'synthetic')
    direct=step(model,optimizer,scaler);expected={k:v.clone() for k,v in model.state_dict().items()};expected_rng=rng_state()
    saved=torch.load(path,map_location='cpu',weights_only=True);restored=STGCNPPActionModel('fight').cuda();restored.load_state_dict(saved['model'])
    opt2=torch.optim.AdamW(restored.parameters(),lr=.001,weight_decay=.0001);opt2.load_state_dict(saved['optimizer'])
    sc2=torch.amp.GradScaler('cuda');sc2.load_state_dict(saved['scaler']);restore_rng(saved['rng'])
    resumed=step(restored,opt2,sc2)
    assert direct==resumed
    assert all(torch.equal(value,restored.state_dict()[key]) for key,value in expected.items())
    for x,y in zip(optimizer.state_dict()['state'].values(),opt2.state_dict()['state'].values()):
        assert all(torch.equal(v,y[k]) if torch.is_tensor(v) else v==y[k] for k,v in x.items())
    checks.append(dict(check='serialized_optimizer_scaler_BN_RNG_resume',next_loss_exact=True,all_model_tensors_exact=True))
    # Repeated inference is stable and does not mutate BN or use training dropout.
    restored.eval();snapshot=tensor_digest(restored.state_dict());x=score_sample(restored,data);y=score_sample(restored,data)
    assert x==y and tensor_digest(restored.state_dict())==snapshot
    empty=dict(items=[{k:v[:0] if k not in ['times'] else v for k,v in data['items'][0].items()}],usable=False)
    assert score_sample(restored,empty) is None
    checks.append(dict(check='evaluation_repeat_and_unknown',exact=True))
    write(OUT/'verification/preflight.json',dict(status='passed',checks=checks,code_hashes={f:sha(ROOT/f) for f in CODE},test_samples_used=0))
    print('PREFLIGHT_PASSED',len(checks),'checks',flush=True)

if __name__=='__main__':main()
