"""Second, denser AI review evidence; no model predictions or label leakage."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scripts.prepare_fine_labels_bc import OUT,read,write,font,read_indices

# Pre-training checks: uncertain/malformed cases and representatives of every
# proposed transition, across all subjects. Sampling alone is not human review.
CHECKS=[
 ('s1_adl_01',.3,2.9),('s1_adl_04',.5,3.5),('s1_adl_11',1.5,4.5),('s1_adl_12',2,5.8),
 ('s2_adl_03',4.3,10),('s2_adl_06',8,10.5),('s2_adl_17',0,4.4),('s2_adl_20',0,6.6),
 ('s3_adl_08',0,2.8),('s3_adl_09',1.7,7.5),('s3_adl_20',7.4,9.38),('s3_adl_21',7.5,10.2),
 ('s4_adl_08',.6,3.7),('s4_adl_14',0,4.1),('s4_adl_15',0,3),('s4_adl_16',.1,4.6),
 ('s1_fall_01',2.7,4.8),('s1_fall_06',2,4.2),('s2_fall_10',1.7,5.1),('s2_fall_11',5.4,7.6),
 ('s2_fall_04',3,6.5),('s3_fall_12',1.2,3.8),('s4_fall_09',1.8,3.3),('s4_fall_15',1.2,3.2),
]


def main():
    lookup={r['id']:r for r in read(OUT/'fine_annotations_draft.json')['videos']}
    for page in range(0,len(CHECKS),6):
        checks=CHECKS[page:page+6]
        sheet=Image.new('RGB',(1680,35+150*len(checks)),'#e9edf1');draw=ImageDraw.Draw(sheet)
        draw.text((5,5),'AI BOUNDARY RECHECK | denser RGB timestamps | NOT independent human truth',fill='black',font=font(20))
        evidence=[]
        for j,(sid,a,b) in enumerate(checks):
            r=lookup[sid];indices=np.linspace(round(a*r['fps']),min(r['frames']-1,round(b*r['fps'])),10).astype(int).tolist()
            frames=read_indices(r['path'],indices);y=35+j*150
            draw.text((3,y),sid,fill='black',font=font(16))
            for i,idx in enumerate(indices):
                im=Image.fromarray(cv2.cvtColor(frames[idx],cv2.COLOR_BGR2RGB));im.thumbnail((166,110))
                sheet.paste(im,(i*168,y+22));draw.text((i*168+3,y+132),f'{idx/r["fps"]:.2f}s',fill='black',font=font(15))
            evidence.append(dict(id=sid,frame_indices=indices))
        name=f'boundary_{page//6+1:02d}'
        sheet.save(OUT/'review'/f'{name}.jpg',quality=93);write(OUT/'review'/f'{name}.json',evidence)
        print(name,flush=True)


if __name__=='__main__':main()
