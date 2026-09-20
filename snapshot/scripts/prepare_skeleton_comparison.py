"""Explicit one-file download, scope review sheets, and immutable derived manifests."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse, collections, json
import cv2, numpy as np
from scripts.skeleton_common import *

def download():
    POSE.parent.mkdir(parents=True,exist_ok=True)
    if not POSE.exists():
        import requests
        temp=POSE.with_suffix('.pt.partial')
        response=requests.get(POSE_URL,stream=True,timeout=(20,120)); response.raise_for_status()
        count=0
        with temp.open('wb') as f:
            for chunk in response.iter_content(1024*128):
                count+=len(chunk)
                if count>POSE_SIZE: raise ValueError('Download exceeded authorized file size')
                f.write(chunk)
        if count!=POSE_SIZE: raise ValueError(f'Incomplete authorized weight: {count}')
        temp.replace(POSE)
    if POSE.stat().st_size!=POSE_SIZE: raise ValueError('Unexpected pose weight size')
    seal(OUT/'pose_provenance.json',dict(path=str(POSE),url=POSE_URL,size=POSE_SIZE,sha256=sha(POSE),
                                      authorization='User authorized this file only in accepted plan'))
    print('pose weight ready',sha(POSE),flush=True)

def strip(row,label):
    cap=cv2.VideoCapture(row['path']); duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/max(1,cap.get(cv2.CAP_PROP_FPS))
    frames=[]
    for t in np.linspace(row.get('start',0),max(0,min(row.get('end',duration),duration)-.08),8):
        cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000);ok,im=cap.read()
        if not ok: im=np.zeros((108,192,3),np.uint8)
        im=cv2.resize(im,(192,108));cv2.putText(im,f'{t:.1f}s',(4,15),0,.4,(0,255,255),1);frames.append(im)
    cap.release(); sheet=np.zeros((132,1536,3),np.uint8)
    cv2.putText(sheet,label[:180],(4,18),0,.5,(255,255,255),1);sheet[24:]=np.hstack(frames)
    return sheet

def sheets():
    folder=OUT/'scope_review';folder.mkdir(parents=True,exist_ok=True)
    g=read(SOURCE/'gmd_manifest.json')['rows']
    selected=[(i,r) for i,r in enumerate(g) if r['label']]
    for offset in range(0,len(selected),8):
        p=folder/f'gmd_{offset//8:02}.jpg'
        if not p.exists(): cv2.imwrite(str(p),np.vstack([strip(r,f'{i} {r["group"]} {Path(r["path"]).name}') for i,r in selected[offset:offset+8]]))
    f=read(SOURCE/'fallvision_recovery_full.json')['rows']; groups=collections.defaultdict(list)
    for r in f:
        if r['label']:groups[r['group']].append(r)
    sampled=[]
    for group,rows in sorted(groups.items()):
        selected=[rows[i] for i in np.linspace(0,len(rows)-1,8).astype(int)]
        p=folder/f'fallvision_{group}.jpg'
        if not p.exists():cv2.imwrite(str(p),np.vstack([strip(r,f'{group} {Path(r["path"]).name}') for r in selected]))
        sampled.extend(selected)
    write(folder/'fallvision_group_samples.json',sampled)
    print('scope review sheets ready',flush=True)

def build():
    review=read(OUT/'scope_review/decisions.json')
    for name,source in [('gmd','gmd_manifest.json'),('tnue','tnue_manifest.json'),('fallvision','fallvision_recovery_full.json'),('vfd','vfd_manifest.json')]:
        original=read(SOURCE/source); accepted=[];excluded=[]
        for i,r0 in enumerate(original['rows']):
            r=dict(r0); reason=None; basis='Inherited source normal label'
            if name=='gmd':
                reason=review['gmd_exclusions'].get(str(i))
                basis='Source action description and eight-frame original-video scope audit'
            elif name=='tnue':
                reason=review['tnue_exclusions'].get(r['id'])
                basis='Second temporal contact-sheet review for visible pushing/striking/grappling; AI provisional'
            elif name=='fallvision' and r['label']:
                if r['group'].startswith('f_raw_b'):
                    reason='Bed-origin group contains bed-surface and floor events; per-video landing scope unverified, quarantined rather than relabelled'
                basis='Source Chair/Stand group label, with stratified landing-scope audit; individual labels not human-verified'
            elif name=='vfd': basis='Inherited source fight/non-fight label; previous exact and edited-source exclusions retained'
            r.update(sample_id=row_id(r),scope_basis=basis,
                     origin=('bed' if 'bed' in r.get('annotation','').lower() else 'chair' if 'chair' in r.get('annotation','').lower() else 'standing_or_other'))
            if name=='fallvision':r['origin']='bed' if '_b_' in r['group'] else 'chair' if '_c_' in r['group'] else 'standing'
            if reason:excluded.append(dict(**r,exclusion_reason=reason));continue
            accepted.append(r)
        isolation(accepted)
        data=dict(dataset=original['dataset'],labels=original['labels'],rows=accepted,exclusions=excluded,
                  parent_manifest=str(SOURCE/source),parent_manifest_sha256=sha(SOURCE/source),
                  scope='floor-directed falls' if name in ('gmd','fallvision') else 'visible pushing or more severe physical aggression',
                  label_status='AI provisional; human acceptance outstanding' if name=='tnue' else 'source labels with documented scope filtering',
                  limitation=review.get(name+'_limitation',''),protocol=PROTOCOL,
                  split_counts={s:dict(collections.Counter(r['label'] for r in accepted if r['split']==s)) for s in sorted({r['split'] for r in accepted})})
        seal(CACHE/'manifests'/f'{name}.json',data)
        print(name,'included',len(accepted),'excluded',len(excluded),data['split_counts'],flush=True)
    seal(OUT/'protocol.json',PROTOCOL)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step',choices=['download','review-sheets','seal']);a=p.parse_args()
    {'download':download,'review-sheets':sheets,'seal':build}[a.step]()
