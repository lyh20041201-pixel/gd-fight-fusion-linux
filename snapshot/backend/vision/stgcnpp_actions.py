"""Pinned 2D ST-GCN++ with shared per-track encoder and task-specific MIL heads."""
from pathlib import Path
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from backend.vision.stgcnpp_backbone import STGCN
from backend.vision.skeleton_actions import prepare_skeleton

WEIGHTS=Path(__file__).resolve().parents[2]/'models/stgcnpp/ntu60_xsub_hrnet_joint.pth'
WEIGHTS_SHA='b274888dd5b7bd8552ce4e3c3073973af84fa0fd18fd94363eac25457f225992'
BACKBONE_CFG=dict(graph_cfg=dict(layout='coco',mode='spatial'),gcn_adaptive='init',gcn_with_res=True,tcn_type='mstcn')

def prepare_stgcnpp(clip,source_shape):
    # Exactly reuse the established eligibility, temporal mask and interaction geometry.
    result=prepare_skeleton(clip,'cpu')
    k=clip['keypoints'].float()
    valid=(k[...,2]>=.3)[:,:,5:].sum(-1)>=4
    indices=torch.as_tensor(clip['frame_indices'])
    usable=torch.tensor([len(torch.unique(indices[v]))>=8 for v in valid],dtype=torch.bool)
    k=k[usable].clone();visible=k[...,2]>=.3
    h,w=source_shape[:2]
    if h<=0 or w<=0:raise ValueError('Invalid source image shape')
    # PYSKL PreNormalize2D(mode=fix), using the true image dimensions.
    k[...,0]=(k[...,0]-w/2)/(w/2)
    k[...,1]=(k[...,1]-h/2)/(h/2)
    k=torch.where(visible[...,None],k,0)
    result['features']=k.permute(0,3,1,2).contiguous()
    assert result['features'].shape[0]==result['valid'].shape[0]
    return result

class STGCNPPActionModel(nn.Module):
    def __init__(self,task):
        super().__init__();self.task=task
        self.backbone=STGCN(**BACKBONE_CFG)
        self.fall_head=nn.Linear(256,1) if task=='fall' else None
        self.pair_head=nn.Sequential(nn.Conv1d(518,128,5,padding=2),nn.ReLU(),nn.Dropout(.2),nn.Conv1d(128,1,3,padding=1)) if task=='fight' else None

    def load_ntu60(self):
        if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()!=WEIGHTS_SHA:raise ValueError('NTU60 weight hash changed')
        weights=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
        state={k.removeprefix('backbone.'):v for k,v in weights.items() if k.startswith('backbone.')}
        excluded=[k for k in weights if not k.startswith('backbone.')]
        if set(excluded)!={'cls_head.fc_cls.weight','cls_head.fc_cls.bias'}:raise ValueError('Unexpected public checkpoint contents')
        self.backbone.load_state_dict(state,strict=True)
        return dict(backbone_tensors_loaded=len(state),excluded=excluded,backbone_parameters=sum(p.numel() for p in self.backbone.parameters()))

    def forward(self,item):
        valid=item['valid'];features=item['features'];m=features.shape[0]
        if m<(2 if self.task=='fight' else 1):return None
        pairs=shared=None
        if self.task=='fight':
            pairs=torch.triu_indices(m,m,offset=1,device=features.device)
            shared=valid[pairs[0]] & valid[pairs[1]]
            good=shared.sum(-1)>=8;pairs=pairs[:,good];shared=shared[good]
            if not pairs.shape[1]:return None
        # M is a variable number of already eligible anonymous tracks. VC BN is
        # the official default and does not require a fixed two-person dimension.
        z=self.backbone(features.permute(0,2,3,1).unsqueeze(0))[0].mean(-1)
        t=z.shape[-1]
        if self.task=='fall':
            mask=F.interpolate(valid[:,None].float(),size=t,mode='nearest')
            pooled=(z*mask).sum(-1)/mask.sum(-1).clamp_min(1)
            return self.fall_head(pooled).squeeze(-1).max()
        outputs=[]
        for pair,mask in zip(pairs.split(128,dim=1),shared.split(128)):
            i,j=pair;scale=((item['scale'][i]+item['scale'][j])/2).clamp_min(1)
            delta=(item['centers'][i]-item['centers'][j])/scale[:,None,None]
            dt=(item['times'][1:]-item['times'][:-1]).clamp_min(1e-6)
            vel=torch.zeros_like(delta);vel[:,1:]=(delta[:,1:]-delta[:,:-1])/dt[None,:,None]
            continuity=mask[:,1:] & mask[:,:-1] & (dt[None]<=.5) & (dt[None]>1e-5)
            vel[:,1:]*=continuity[:,:,None]
            rel=torch.cat([delta.abs(),delta.square().sum(-1,keepdim=True).sqrt(),vel.abs(),mask[:,:,None]],-1)
            rel=F.interpolate((rel*mask[:,:,None]).permute(0,2,1),size=t,mode='linear',align_corners=False)
            pair_features=torch.cat([(z[i]+z[j])/2,(z[i]-z[j]).abs(),rel],1)
            pred=self.pair_head(pair_features).squeeze(1)
            down=F.interpolate(mask[:,None].float(),size=t,mode='nearest').squeeze(1)
            outputs.append((pred*down).sum(-1)/down.sum(-1).clamp_min(1))
        return torch.cat(outputs).max()
