"""Audit failed source tails and prepare a separately traceable recovery subset."""
from pathlib import Path
import argparse,json,cv2,hashlib
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output-prefix',required=True);a=p.parse_args()
    run=json.loads(Path(a.evaluation).read_text());data=json.loads(Path(a.manifest).read_text());lookup={r['path']:r for r in data['rows']}
    corrections=[];rows=[];unrecoverable=[]
    for error in run['failures']:
        row=dict(lookup[error['path']]);cap=cv2.VideoCapture(row['path']);fps=cap.get(cv2.CAP_PROP_FPS);declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release()
        audit=dict(path=row['path'],declared_frames=declared,decoded_frames=count,fps=fps)
        if fps<=0 or count<=0 or not 0<declared-count<=3:
            unrecoverable.append(audit);continue
        row['source_declared_duration']=row['duration'];row['duration']=count/fps
        row['duration_correction']='Use actual sequentially decodable frame count; source header overstates tail by 1-3 frames'
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
        corrections.append(audit);rows.append(row)
    output=Path(a.output_prefix)
    recovery=dict(dataset=data['dataset'],labels=data['labels'],rows=rows,
                  protocol='Recovery subset only; source files and labels unchanged; actual decoded duration corrects overstated source metadata',
                  corrections=corrections,unrecoverable=unrecoverable)
    output.with_suffix('.json').write_text(json.dumps(recovery,indent=2))
    lookup.update({r['path']:r for r in rows});full=dict(data);full['rows']=[lookup[r['path']] for r in data['rows']];full['duration_corrections']=corrections
    output.with_name(output.name+'_full.json').write_text(json.dumps(full,indent=2))
    print('recoverable',len(rows),'unrecoverable',len(unrecoverable))

if __name__=='__main__':main()
