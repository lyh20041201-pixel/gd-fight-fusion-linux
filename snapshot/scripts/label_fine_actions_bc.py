"""Recorded AI review of GMD RGB sheets, independent of any action predictions.

Numbers below are approximate transition boundaries from the RGB review,
anchored to the original publisher's descriptions. They are NOT human truth.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import re
from collections import Counter
from scripts.prepare_fine_labels_bc import ROOT, OUT, read, sha, seal, write

CLASSES=['fall','rising','sitting_down','bending','lying_down','seated','lying','other']

# Explicit RGB-reviewed transition spans. Rising means getting up from a seated
# or lying support, not merely straightening the torso after bending.
TRANSITIONS={
 's1_adl_01':[(.6,2.6,'lying_down')], 's1_adl_02':[(.4,2,'lying_down')],
 's1_adl_03':[(.4,1.8,'lying_down')], 's1_adl_04':[(.9,3,'rising')],
 's1_adl_05':[(2.5,4,'sitting_down')], 's1_adl_11':[(1.7,3.2,'bending'),(3.2,4.1,'sitting_down')],
 's1_adl_12':[(2.6,3.6,'rising'),(4.1,5.5,'bending')],
 's1_adl_13':[(2.6,3.9,'sitting_down')], 's1_adl_14':[(.5,1.5,'rising')],
 's1_adl_15':[(1.2,3,'bending'),(3.3,4.1,'sitting_down')], 's1_adl_16':[(.4,1.9,'bending')],
 's2_adl_01':[(6.7,7.7,'rising')], 's2_adl_02':[(.7,3.7,'bending')],
 's2_adl_03':[(4.6,5.8,'sitting_down'),(7.3,9.3,'lying_down')],
 's2_adl_04':[(.8,5.3,'bending')], 's2_adl_05':[(2.3,3.8,'rising')],
 's2_adl_06':[(1.7,4.1,'bending'),(8.6,9.8,'sitting_down')],
 's2_adl_08':[(.9,4.4,'bending'),(6.5,10.4,'bending')],
 's2_adl_09':[(.2,1.3,'rising'),(3.1,4.5,'sitting_down')],
 's2_adl_10':[(0,1.3,'rising')], 's2_adl_12':[(.8,3.7,'lying_down')],
 's2_adl_13':[(.5,1.9,'sitting_down')], 's2_adl_14':[(3.5,5.3,'sitting_down')],
 's2_adl_16':[(8.5,10.1,'lying_down')],
 's2_adl_17':[(.4,2.2,'rising'),(2.7,4,'rising')],
 's2_adl_19':[(.5,2.8,'sitting_down')],
 's2_adl_20':[(0,1.7,'rising'),(3.2,6.1,'sitting_down')],
 's2_adl_21':[(2.3,6,'bending')], 's2_adl_22':[(8.4,10.7,'sitting_down')],
 's3_adl_03':[(1.6,3.3,'rising'),(7.3,8.3,'rising')],
 's3_adl_04':[(.5,2.7,'bending'),(5.1,6.8,'bending'),(8.7,9.7,'sitting_down')],
 's3_adl_05':[(0,.9,'bending'),(1.6,3.8,'bending'),(4.5,5.17,'bending')],
 's3_adl_06':[(0,2.6,'bending'),(3.2,4.9,'bending'),(4.9,6,'sitting_down')],
 's3_adl_08':[(.2,2.3,'lying_down')],
 's3_adl_09':[(2,3.7,'rising'),(5.8,7.2,'rising')],
 's3_adl_12':[(0,1,'sitting_down'),(8.1,9.3,'rising')],
 's3_adl_13':[(0,1.4,'sitting_down')], 's3_adl_14':[(0,1.6,'sitting_down')],
 's3_adl_18':[(.5,1.7,'sitting_down')],
 's3_adl_19':[(0,1.7,'bending'),(2.7,4.6,'bending')],
 's3_adl_21':[(8.3,9.4,'bending')],
 's4_adl_08':[(.9,3,'rising'),(4.7,7.1,'bending')],
 's4_adl_10':[(1.7,3.1,'sitting_down'),(3.1,4.8,'lying_down')],
 's4_adl_14':[(0,.9,'rising'),(2.1,3.7,'bending')],
 's4_adl_15':[(.3,2.5,'lying_down')],
 's4_adl_16':[(.4,1.4,'sitting_down'),(2.7,4.3,'lying_down')],
 's4_adl_17':[(1.1,3.4,'bending')], 's4_adl_18':[(.3,1.2,'sitting_down')],
 's4_adl_19':[(1.3,3.3,'bending'),(6.5,7.3,'sitting_down')],
 's4_adl_20':[(0,1.5,'bending'),(2.5,3.5,'sitting_down'),(14.5,16.5,'bending')],
 # Normal actions inside fall videos are also negative supervision.
 's2_fall_07':[(.8,1.9,'bending')], 's2_fall_11':[(1.7,3.4,'lying_down')],
 's2_fall_13':[(0,1.4,'bending')], 's2_fall_18':[(0,2.2,'bending')],
 's2_fall_23':[(0,3.4,'bending')], 's2_fall_24':[(1.8,2.9,'rising')],
 's2_fall_25':[(0,2.3,'bending')],
}

# Stable support overrides (whole video initially other, these replace it, then
# transitions/fall spans take precedence). -1 means actual source duration.
STATES={
 's1_adl_01':[(0,.6,'seated'),(2.6,-1,'lying')],
 's1_adl_02':[(0,.4,'seated'),(2,-1,'lying')],
 's1_adl_03':[(0,.4,'seated'),(1.8,-1,'lying')],
 's1_adl_04':[(0,.9,'lying'),(3,-1,'seated')],
 's1_adl_05':[(4,-1,'seated')], 's1_adl_06':[(0,-1,'seated')],
 's1_adl_07':[(0,-1,'seated')], 's1_adl_11':[(4.1,-1,'seated')],
 's1_adl_12':[(0,2.6,'seated')], 's1_adl_13':[(3.9,-1,'seated')],
 's1_adl_14':[(0,.5,'seated')], 's1_adl_15':[(4.1,-1,'seated')], 's1_adl_16':[(0,-1,'seated')],
 's2_adl_01':[(0,6.7,'seated')], 's2_adl_02':[(0,-1,'seated')],
 's2_adl_03':[(5.8,7.3,'seated'),(9.3,-1,'lying')], 's2_adl_05':[(0,2.3,'seated')],
 's2_adl_06':[(9.8,-1,'seated')], 's2_adl_09':[(0,.2,'seated'),(4.5,-1,'seated')],
 's2_adl_12':[(3.7,-1,'lying')], 's2_adl_13':[(1.9,-1,'seated')],
 's2_adl_14':[(5.3,-1,'seated')], 's2_adl_15':[(0,-1,'seated')],
 's2_adl_16':[(0,8.5,'seated'),(10.1,-1,'lying')],
 's2_adl_17':[(0,.4,'lying'),(2.2,2.7,'seated')],
 's2_adl_19':[(2.8,-1,'seated')], 's2_adl_20':[(6.1,-1,'seated')],
 's2_adl_22':[(10.7,-1,'seated')], 's2_adl_23':[(0,-1,'seated')],
 's3_adl_01':[(0,-1,'seated')], 's3_adl_03':[(0,1.6,'lying'),(3.3,7.3,'seated')],
 's3_adl_04':[(9.7,-1,'seated')], 's3_adl_06':[(6,-1,'seated')],
 's3_adl_08':[(0,.2,'seated'),(2.3,-1,'lying')],
 's3_adl_09':[(0,2,'lying'),(3.7,5.8,'seated')], 's3_adl_12':[(1,8.1,'seated')],
 's3_adl_13':[(1.4,-1,'seated')], 's3_adl_14':[(1.6,-1,'seated')],
 's3_adl_15':[(0,-1,'seated')], 's3_adl_16':[(0,-1,'seated')], 's3_adl_17':[(0,-1,'seated')],
 's3_adl_18':[(1.7,-1,'seated')], 's3_adl_20':[(0,8.5,'seated'),(8.5,-1,'uncertain')],
 's3_adl_21':[(0,-1,'seated')], 's3_adl_22':[(0,-1,'seated')],
 's4_adl_01':[(0,-1,'seated')], 's4_adl_03':[(0,-1,'seated')], 's4_adl_04':[(0,-1,'seated')],
 's4_adl_08':[(0,.9,'lying'),(3,-1,'seated')], 's4_adl_09':[(0,-1,'seated')],
 's4_adl_10':[(4.8,-1,'lying')], 's4_adl_11':[(0,-1,'seated')], 's4_adl_12':[(0,-1,'seated')],
 's4_adl_13':[(0,-1,'seated')], 's4_adl_15':[(0,.3,'seated'),(2.5,-1,'lying')],
 's4_adl_16':[(1.4,2.7,'seated'),(4.3,-1,'lying')], 's4_adl_18':[(1.2,-1,'seated')],
 's4_adl_19':[(7.3,-1,'seated')], 's4_adl_20':[(3.5,-1,'seated')],
}

FALL_END={
 1:[4.4,2.8,3.5,3.2,3,3.8,2.3,2.1,1.9,2,2.6,3.5,1.5,1.9,3.2,4],
 2:[3.4,4.5,1.2,6,2.9,4,2.9,4.2,3.6,4.8,7.1,8.8,5.1,4.7,8.8,1.8,3.6,11.1,9.6,9.6,7.3,2.8,4.8,4.7,4.6],
 3:[1.4,3.8,3,2.5,4.8,.8,2.3,1.6,4.3,2.5,6.5,3.3,2.1,9.6,3.3,3,2.5,5.5,3.3,3.5,6.8],
 4:[2.8,2.5,2.8,2.2,3.6,4,3.2,3.1,2.9,3.1,2.2,2.7,1.4,3.4,2.9,4.2,3.7],
}


# Changes made after viewing the 24 denser boundary montages, before fitting.
TRANSITIONS.update({
 's1_adl_04':[(.9,3.4,'rising')],
 's2_adl_03':[(4.4,5.5,'sitting_down'),(8.1,9.8,'lying_down')],
 's2_adl_17':[(1.6,3,'rising'),(3.1,3.9,'rising')],
 's2_adl_20':[(0,.9,'rising'),(1.1,2.8,'bending'),(3.4,6.4,'sitting_down')],
 's3_adl_08':[(.9,2.4,'lying_down')],
 's3_adl_09':[(2.1,3.3,'rising'),(6,7.4,'rising')],
 's4_adl_14':[(0,.5,'rising'),(1.7,3.6,'bending')],
 's4_adl_16':[(.4,1.2,'sitting_down'),(2.7,3.8,'lying_down')],
})
STATES.update({
 's1_adl_04':[(0,.9,'lying'),(3.4,-1,'seated')],
 's2_adl_03':[(5.5,8.1,'seated'),(9.8,-1,'lying')],
 's2_adl_17':[(0,1.6,'lying'),(3,3.1,'seated')],
 's2_adl_20':[(6.4,-1,'seated')],
 's3_adl_08':[(0,.9,'seated'),(2.4,-1,'lying')],
 's3_adl_09':[(0,2.1,'lying'),(3.3,6,'seated')],
 's3_adl_20':[(0,-1,'seated')],
 's4_adl_16':[(1.2,2.7,'seated'),(3.8,-1,'lying')],
})
FALL_END[2][9]=4.3


def build():
    rows=read(OUT/'source_inventory.json')['videos']
    original=read(ROOT/'datasets/video_events/rebuild/gmd_manifest.json')['rows']
    output=[]
    for row in rows:
        sid=row['id']; duration=row['duration']; layers=[(0,duration,'other')]
        # Publisher's posture intervals supplement the reviewed stable spans.
        for name, label in [('Sitting','seated'),('Sleeping','lying')]:
            for content in re.findall(name+r'\s*\[([^\]]+)\]',row['annotation'],re.I):
                for a,b in re.findall(r'([\d.]+)\s*to\s*([\d.]+)',content):
                    a,b=float(a),min(duration,float(b))
                    if 0<=a<b:layers.append((a,b,label))
        layers.extend((a,duration if b==-1 else min(b,duration),c) for a,b,c in STATES.get(sid,[]))
        layers.extend((a,min(b,duration),c) for a,b,c in TRANSITIONS.get(sid,[]))
        events=[]
        if row['category']=='Fall':
            source_ranges=[(r['start'],r['end']) for r in original if r['path']==row['path']]
            assert source_ranges
            # Adjacent directional components in s2_fall_10 describe one fall.
            onset=min(a for a,b in source_ranges)
            end=FALL_END[int(row['group'][-1])][int(sid[-2:])-1]
            assert onset<end<=duration, sid
            layers.extend([(end,duration,'lying'),(onset,end,'fall')])
            events=[dict(start=onset,dynamic_end=end,source_intervals=source_ranges,
                         onset_provenance='prior_AI_correction' if sid=='s4_fall_15' else 'publisher_timestamp',
                         dynamic_end_provenance='AI_RGB_review_approximate')]
        bounds=sorted({0.,duration,*[float(v) for a,b,c in layers for v in (a,b)]})
        spans=[]
        for a,b in zip(bounds,bounds[1:]):
            if not 0<=a<b<=duration:raise ValueError(sid)
            mid=(a+b)/2
            label=next(c for x,y,c in reversed(layers) if x<=mid<y)
            if spans and spans[-1]['label']==label:spans[-1]['end']=b
            else:spans.append(dict(start=a,end=b,label=label))
        page=(int(sid[-2:])-1)//6+1
        evidence=OUT/'review'/f"s{row['group'][-1]}_{row['category'].lower()}_{page:02d}.jpg"
        assert evidence.exists()
        output.append(dict(**row,spans=spans,events=events,
            annotation_provenance='AI refined source labels; RGB temporal montage reviewed, not independent human confirmation',
            human_confirmed=False,boundary_uncertainty_seconds=.35,
            review_evidence=str(evidence),review_evidence_sha256=sha(evidence),
            person_scope='sole intended actor; anonymous detector track fragments; no cross-person identity inference',
            quality_flags=[x for x,yes in [('night', 'Night' in row['annotation']),
                ('partial_body',any(w in row['annotation'].lower() for w in ['partial','not visible','waist to toe'])),
                ('source_timestamp_correction',bool(row['correction']))] if yes]))
    assert all(any(s['label']=='fall' for s in r['spans'])==(r['category']=='Fall') for r in output)
    result=dict(classes=CLASSES,version=1,reviewer='Codex AI',human_review=False,
        review_scope='All 160 ten-frame RGB source montages inspected; transition boundary rechecks recorded separately. No frame-perfect annotation claim.',
        label_rules=dict(fall='source fall event, dynamic descent/toppling only; post-fall lying separate',
                         rising='lying or seated to higher support; straightening after bending remains bending',
                         other='walking, standing, push-ups/squats and other normal actions; not mislabeled as deliberate lying',
                         lying_down='deliberate descent into lying according to source ADL description',
                         uncertain='excluded from fine training'),
        videos=output,split_counts=dict(Counter(r['split'] for r in output)))
    write(OUT/'fine_annotations_draft.json',result)
    print('DRAFT',len(output),result['split_counts'])


if __name__=='__main__':build()
