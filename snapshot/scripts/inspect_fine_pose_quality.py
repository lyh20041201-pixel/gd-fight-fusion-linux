"""Input-only diagnostics, without reading test model predictions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import Counter,defaultdict
import numpy as np
import cv2
from PIL import Image,ImageDraw
from scripts.prepare_fine_labels_bc import OUT,CACHE,read,write,font,read_indices


def main():
    rows={r['id']:r for r in read(OUT/'fine_annotations.json')['videos']}
    windows=read(CACHE/'windows.json')['windows'];groups=defaultdict(list)
    for w in windows:groups[w['video_id']].append(w)
    report=[]
    for sid,ws in groups.items():
        usable=[w for w in ws if w['feature_index']>=0]
        report.append(dict(id=sid,split=rows[sid]['split'],category=rows[sid]['category'],
            usable_windows=len(usable),total_windows=len(ws),
            min_primary_coverage=min((w['visible_frame_fraction'] for w in usable),default=None),
            fragmented_windows=sum(w['eligible_tracks']>1 for w in usable),
            low_primary_coverage_windows=sum(w['visible_frame_fraction']<.5 for w in usable),
            reasons=dict(Counter(w['reason'] for w in ws if w['reason']))))
    write(OUT/'pose_input_audit.json',dict(rows=report,
        usable_windows=sum(r['usable_windows'] for r in report),
        multiple_eligible_track_fragments=sum(r['fragmented_windows'] for r in report),
        low_primary_coverage_windows=sum(r['low_primary_coverage_windows'] for r in report),
        note='Multiple track fragments within a 4s history do not necessarily mean multiple simultaneous people. No prediction-based sample selection.'))
    # Lowest coverage among each subject's fall and normal usable windows.
    chosen=[]
    for subject in range(1,5):
        for category in ['ADL','Fall']:
            candidates=[w for w in windows if rows[w['video_id']]['group']==f'subject-{subject}'
                        and rows[w['video_id']]['category']==category and w['feature_index']>=0]
            chosen.append(min(candidates,key=lambda w:w['visible_frame_fraction']))
    sheet=Image.new('RGB',(1680,len(chosen)*160+35),'#e9edf1');draw=ImageDraw.Draw(sheet)
    draw.text((6,5),'POSE INPUT AUDIT | lowest primary-track coverage per subject/category | all detector joints shown',fill='black',font=font(19))
    for j,w in enumerate(chosen):
        row=rows[w['video_id']];z=np.load(CACHE/'pose'/f'{row["id"]}.npz')
        src_indices=np.asarray(w['sampled_frame_indices'])[np.linspace(0,31,10).astype(int)]
        frames=read_indices(row['path'],src_indices.tolist());y=35+j*160
        draw.text((5,y),f"{row['id']}  end={w['end']:.2f}s  primary ID {w['primary_track']}  valid={w['visible_frame_fraction']:.0%}  fragments={w['eligible_tracks']}",fill='black',font=font(16))
        for i,idx in enumerate(src_indices):
            frame=frames[int(idx)].copy();t=int(np.where(z['frame_indices']==idx)[0][0])
            for track,tid in enumerate(z['track_ids']):
                joints=z['keypoints'][track,t];color=(40,220,30) if tid==w['primary_track'] else (0,110,255)
                for x,yy,c in joints:
                    if c>=.3:cv2.circle(frame,(round(x),round(yy)),3,color,-1)
                visible=joints[joints[:,2]>=.3]
                if len(visible):cv2.putText(frame,str(tid),(int(visible[:,0].min()),max(15,int(visible[:,1].min())-4)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
            im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((166,115));sheet.paste(im,(i*168,y+23))
            draw.text((i*168+3,y+140),f'{idx/row["fps"]:.2f}s',fill='black',font=font(14))
    sheet.save(OUT/'review'/'pose_quality.jpg',quality=93)
    print('POSE_DIAGNOSTICS',sum(r['fragmented_windows'] for r in report),'fragmented windows',flush=True)


if __name__=='__main__':main()
