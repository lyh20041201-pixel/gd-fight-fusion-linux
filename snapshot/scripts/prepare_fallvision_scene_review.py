"""Arrange existing RGB audit thumbnails for exhaustive scene review; no labels inferred."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import cv2,numpy as np
from scripts.skeleton_common import read,write,seal,sha,digest,offline,CACHE
from scripts.skeleton_round2 import OUT


def scene_image(row):
    image=cv2.imread(str(OUT/'source_review/thumbs/fallvision'/(row['sample_id']+'.jpg')))
    frames=np.stack(np.split(image,3,axis=1));image=np.median(frames,axis=0).astype(np.uint8)
    # Remove letterbox for source comparison only; action-model pixels never change.
    mask=(image.max(2)>18);ys,xs=np.nonzero(mask)
    if len(ys):image=image[ys.min():ys.max()+1,xs.min():xs.max()+1]
    return cv2.resize(image,(112,112))


def main():
    offline();cv2.setNumThreads(3);cv2.setRNGSeed(42)
    folder=OUT/'source_review/fallvision_scenes_v2';folder.mkdir(parents=True,exist_ok=True)
    rows=read(CACHE/'manifests/fallvision.json')['rows']
    if (folder/'clusters.json').exists():print('already arranged',flush=True);return
    images=[scene_image(r) for r in rows]
    vectors=np.stack([cv2.resize(im,(16,16)).reshape(-1).astype(np.float32)/255 for im in images])
    _,labels,centers=cv2.kmeans(vectors,96,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,40,.002),1,cv2.KMEANS_PP_CENTERS)
    clusters=[]
    for cluster in range(96):
        ids=np.flatnonzero(labels[:,0]==cluster).tolist()
        ids.sort(key=lambda i:float(np.square(vectors[i]-centers[cluster]).sum()))
        examples=[ids[k] for k in sorted(set(np.linspace(0,len(ids)-1,min(6,len(ids))).astype(int)))]
        pages=[]
        for start in range(0,len(ids),100):
            part=ids[start:start+100];canvas=np.full((((len(part)+9)//10)*132,1120,3),24,np.uint8)
            members=[]
            for j,i in enumerate(part):
                y=(j//10)*132;x=(j%10)*112;canvas[y+20:y+132,x:x+112]=images[i]
                cv2.putText(canvas,f'{start+j:03d} {rows[i]["sample_id"][:6]}',(x+2,y+14),0,.32,(255,255,255),1)
                members.append(dict(position=start+j,sample_id=rows[i]['sample_id'],path=rows[i]['path']))
            target=folder/f'cluster_{cluster:02d}_{start//100}.jpg';cv2.imwrite(str(target),canvas)
            pages.append(dict(path=str(target),members=members))
        clusters.append(dict(cluster=cluster,count=len(ids),pages=pages,
            examples=[dict(sample_id=rows[i]['sample_id'],path=rows[i]['path']) for i in examples],
            ordered_sample_ids=[rows[i]['sample_id'] for i in ids]))
    for start in range(0,96,8):
        canvas=np.full((8*140,784,3),24,np.uint8)
        for j,c in enumerate(clusters[start:start+8]):
            cv2.putText(canvas,f'Cluster {c["cluster"]:02d}  n={c["count"]}',(5,j*140+17),0,.45,(255,255,255),1)
            for k,e in enumerate(c['examples']):
                i=next(i for i,r in enumerate(rows) if r['sample_id']==e['sample_id'])
                canvas[j*140+24:j*140+136,k*128:k*128+112]=images[i]
        cv2.imwrite(str(folder/f'overview_{start//8:02d}.jpg'),canvas)
    seal(folder/'clusters.json',dict(status='layout_only_no_source_assignments',method='RGB thumbnail median, black margin trim, 16x16 spatial color kmeans96 seed42; grouping is only for review layout',
        source_manifest_sha256=sha(CACHE/'manifests/fallvision.json'),rows=len(rows),clusters=clusters,
        label_or_model_predictions_used=False))
    print('scene review prepared',len(rows),'samples in',96,'layout clusters',flush=True)


if __name__=='__main__':main()
