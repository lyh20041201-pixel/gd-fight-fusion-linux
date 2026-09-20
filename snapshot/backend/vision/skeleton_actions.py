"""Small ST-GCN and symmetric pair interaction head; offline research model."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

EDGES=[(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(7,9),
       (6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

def adjacency():
    a=torch.zeros(3,17,17);a[0]=torch.eye(17)
    for u,v in EDGES:a[1,u,v]=1;a[2,v,u]=1
    return a/a.sum(1,keepdim=True).clamp_min(1)

class GraphBlock(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__();self.cout=cout
        self.graph=nn.Conv2d(cin,cout*3,1)
        self.temporal=nn.Sequential(nn.GroupNorm(8,cout),nn.ReLU(),
            nn.Conv2d(cout,cout,(9,1),(stride,1),(4,0)),nn.GroupNorm(8,cout),nn.Dropout(.2))
        self.residual=nn.Identity() if cin==cout and stride==1 else nn.Conv2d(cin,cout,1,(stride,1))
    def forward(self,x,a):
        n,_,t,v=x.shape;z=self.graph(x).reshape(n,3,self.cout,t,v)
        z=torch.einsum('nkctv,kvw->nctw',z,a)
        return F.relu(self.temporal(z)+self.residual(x))

class SkeletonActionModel(nn.Module):
    def __init__(self,task):
        super().__init__();self.task=task;self.register_buffer('adjacency',adjacency())
        channels=[6,32,32,64,64,128,128];strides=[1,1,2,1,2,1]
        self.blocks=nn.ModuleList([GraphBlock(channels[i],channels[i+1],strides[i]) for i in range(6)])
        self.fall_head=nn.Linear(128,1)
        self.pair_head=nn.Sequential(nn.Conv1d(262,128,5,padding=2),nn.ReLU(),nn.Dropout(.2),nn.Conv1d(128,1,3,padding=1))

    def forward(self,item):
        x=item['features'];valid=item['valid']
        if x.shape[0] < (2 if self.task=='fight' else 1):return None
        for block in self.blocks:x=block(x,self.adjacency)
        x=x.mean(-1);t=x.shape[-1]
        if self.task=='fall':
            mask=F.interpolate(valid[:,None].float(),size=t,mode='nearest')
            pooled=(x*mask).sum(-1)/mask.sum(-1).clamp_min(1)
            return self.fall_head(pooled).squeeze(-1).max()
        pairs=torch.triu_indices(x.shape[0],x.shape[0],offset=1,device=x.device)
        shared=valid[pairs[0]] & valid[pairs[1]]
        good=shared.sum(-1)>=8
        pairs=pairs[:,good];shared=shared[good]
        if not pairs.shape[1]:return None
        outputs=[]
        for pair,mask in zip(pairs.split(128,dim=1),shared.split(128)):
            i,j=pair;scale=((item['scale'][i]+item['scale'][j])/2).clamp_min(1)
            delta=(item['centers'][i]-item['centers'][j])/scale[:,None,None]
            dt=(item['times'][1:]-item['times'][:-1]).clamp_min(1e-6)
            vel=torch.zeros_like(delta);vel[:,1:]=(delta[:,1:]-delta[:,:-1])/dt[None,:,None]
            continuity=mask[:,1:] & mask[:,:-1] & (dt[None]<=.5) & (dt[None]>1e-5)
            vel[:,1:]*=continuity[:,:,None]
            rel=torch.cat([delta.abs(),delta.square().sum(-1,keepdim=True).sqrt(),vel.abs(),mask[:,:,None]],-1)
            rel=rel*mask[:,:,None]
            rel=F.interpolate(rel.permute(0,2,1),size=t,mode='linear',align_corners=False)
            pair_features=torch.cat([(x[i]+x[j])/2,(x[i]-x[j]).abs(),rel],1)
            pred=self.pair_head(pair_features).squeeze(1)
            down=F.interpolate(mask[:,None].float(),size=t,mode='nearest').squeeze(1)
            outputs.append((pred*down).sum(-1)/down.sum(-1).clamp_min(1))
        return torch.cat(outputs).max()

def prepare_skeleton(clip,device='cpu'):
    """Fixed-window anchors preserve descent; missing/duplicated observations have no velocity."""
    k=clip['keypoints'].float();boxes=clip['boxes'].float();times=torch.tensor(clip['timestamps'],dtype=torch.float32)
    valid=(k[...,2]>=.3)[:,:,5:].sum(-1)>=4
    indices=torch.tensor(clip['frame_indices'])
    usable=torch.tensor([len(torch.unique(indices[v]))>=8 for v in valid],dtype=torch.bool)
    k=k[usable];boxes=boxes[usable];valid=valid[usable]
    centers=(boxes[...,:2]+boxes[...,2:])/2
    scales=[];anchors=[]
    for b,c,v in zip(boxes,centers,valid):
        scales.append(torch.linalg.vector_norm(b[v,2:]-b[v,:2],dim=-1).median().clamp_min(1))
        anchors.append(c[v].median(0).values)
    scale=torch.stack(scales) if scales else torch.empty(0)
    anchor=torch.stack(anchors) if anchors else torch.empty(0,2)
    visible=k[...,2]>=.3
    xy=(k[...,:2]-anchor[:,None,None,:])/scale[:,None,None,None]
    xy=torch.where(visible[...,None],xy,0)
    velocity=torch.zeros_like(xy);dt=times[1:]-times[:-1]
    continuity=visible[:,1:] & visible[:,:-1] & (dt[None,:,None]>.00001) & (dt[None,:,None]<=.5)
    velocity[:,1:]=(xy[:,1:]-xy[:,:-1])/dt.clamp_min(1e-6)[None,:,None,None]
    velocity[:,1:]=torch.where(continuity[...,None],velocity[:,1:],0)
    features=torch.cat([xy,velocity.clamp(-20,20),k[...,2:3]*visible[...,None],visible[...,None].float()],-1).permute(0,3,1,2)
    return {name:value.to(device) for name,value in dict(features=features,valid=valid,centers=centers,scale=scale,times=times).items()}
