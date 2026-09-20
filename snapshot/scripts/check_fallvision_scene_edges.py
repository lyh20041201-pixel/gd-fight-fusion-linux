"""Offline cross-scene similarity and canonical filename review evidence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import re
from collections import defaultdict,Counter
import cv2,numpy as np
from scripts.skeleton_common import read,write,sha,offline
from scripts.skeleton_round2 import OUT

FOLDER=OUT/'source_review/fallvision_scenes_v2'

def main():
    offline();cv2.setNumThreads(2)
    layout=read(FOLDER/'clusters.json');review=read(FOLDER/'ai_review_working.json')
    assert review['reviewed_clusters']==list(range(96))
    rows=[];names=defaultdict(list)
    for cluster in layout['clusters']:
        rule=review['assignments'].get(str(cluster['cluster']),{})
        for page in cluster['pages']:
            for member in page['members']:
                group=rule.get('default',review['default'])
                for key,values in rule.items():
                    if key!='default' and member['position'] in values:group=key
                row=dict(**member,scene=group,cluster=cluster['cluster'],evidence=page['path'])
                row['canonical_basename']=re.sub(r'(?i)_resized','',Path(row['path']).stem)
                names[row['canonical_basename']].append(row);rows.append(row)
    # Ambiguity propagates to every canonical-name variant before any splitting.
    for members in names.values():
        if any(r['scene']=='quarantine' for r in members):
            for r in members:r['scene']='quarantine'
        if len({r['scene'] for r in members})>1:raise ValueError('Cross-scene canonical filename relation: '+str(members))
    hashes=[];refs=[]
    for r in rows:
        im=cv2.imread(str(OUT/'source_review/thumbs/fallvision'/(r['sample_id']+'.jpg')))
        for index,frame in enumerate(np.split(im,3,axis=1)):
            gray=cv2.cvtColor(cv2.resize(frame,(32,32)),cv2.COLOR_BGR2GRAY).astype(np.float32)
            dct=cv2.dct(gray)[:8,:8].reshape(-1);bits=dct>np.median(dct[1:])
            hashes.append(np.packbits(bits).view('>u8')[0]);refs.append((r,index))
    hashes=np.array(hashes,dtype=np.uint64);groups=np.array([x[0]['scene'] for x in refs]);pairs={}
    pop=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
    for i,(row,frame) in enumerate(refs):
        if row['scene']=='quarantine':continue
        candidates=np.flatnonzero((groups[i+1:]!=row['scene'])&(groups[i+1:]!='quarantine'))+i+1
        if not len(candidates):continue
        dist=pop[np.bitwise_xor(hashes[i],hashes[candidates]).view(np.uint8).reshape(-1,8)].sum(1)
        for offset in np.flatnonzero(dist<=12):
            j=int(candidates[offset]);other,other_frame=refs[j];key=tuple(sorted([row['sample_id'],other['sample_id']]))
            if key not in pairs or dist[offset]<pairs[key]['distance']:
                pairs[key]=dict(a=row['sample_id'],b=other['sample_id'],frame_a=frame,frame_b=other_frame,distance=int(dist[offset]),scene_a=row['scene'],scene_b=other['scene'])
    pairs=sorted(pairs.values(),key=lambda r:(r['distance'],r['a'],r['b']))
    result=dict(rows=rows,counts=dict(Counter(r['scene'] for r in rows)),cross_group_candidates=pairs,
        method='Canonical basename removes only _resized; all three cached RGB thumbnail frames; 64-bit DCT pHash Hamming<=12 across proposed groups. Similarity is candidate evidence, not duplicate proof.',
        limitation='Three cached timestamps and AI scene review cannot rule out undiscovered transformations or common participants. This supports scene containment only.',
        review_sha256=sha(FOLDER/'ai_review_working.json'))
    write(FOLDER/'source_edges_working.json',result)
    by={r['sample_id']:r for r in rows}
    for start in range(0,len(pairs),12):
        part=pairs[start:start+12];canvas=np.full((len(part)*140,760,3),24,np.uint8)
        for n,p in enumerate(part):
            cv2.putText(canvas,f'{start+n:03d} d={p["distance"]} {p["scene_a"]} / {p["scene_b"]}',(4,n*140+16),0,.4,(255,255,255),1)
            for k,key in enumerate(['a','b']):
                im=cv2.imread(str(OUT/'source_review/thumbs/fallvision'/(p[key]+'.jpg')))
                canvas[n*140+23:n*140+135,k*380:k*380+336]=im
        cv2.imwrite(str(FOLDER/f'cross_scene_{start//12:03d}.jpg'),canvas)
    print('SCENE COUNTS',result['counts'],'CROSS SCENE CANDIDATES',len(pairs),flush=True)

if __name__=='__main__':main()
