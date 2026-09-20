"""Seal explicitly reviewed AI labels; human acceptance remains outstanding."""
from pathlib import Path
from collections import Counter
import json,random

ROOT=Path(__file__).resolve().parents[1]
REJECT={
'00_1':'Weapon threat, no unambiguous physical fight in selected window',
'02_0':'Chair raised and conflict starts inside proposed normal window',
'05_0':'Physical conflict starts inside proposed normal window',
'07_1':'Standing interaction; fight not unambiguous',
'08_0':'Person dragged/lying on floor; ambiguous aftermath',
'13_1':'Restraint/argument; no unambiguous fight in selected window',
'14_1':'Approach/restraint; no unambiguous blows in selected window',
'15_1':'Distant gestures, physical contact unclear',
'17_1':'People obstruct the relevant action; no clear fight visible',
'24_1':'Running/chasing dominates; contact not clear',
'26_1':'Standing interaction; physical aggression unclear',
'28_1':'Aftermath only; person on floor without new visible aggression',
'29_1':'Standing crowd; contact ambiguous at this resolution',
'31_0':'Confrontation begins near end of normal candidate',
'34_1':'Driver exits vehicle; selected window lacks clear fighting',
'35_1':'Crowd restraint/shoving ambiguous',
'37_1':'Target obscured by people and bushes',
'40_1':'Distant interaction; action cannot be confidently resolved',
'45_0':'Guarding/raised arms at conflict boundary',
'46_0':'Repeated news freeze frame, unsuitable natural negative',
'58_1':'Counter crossing; attack versus theft unclear',
'61_0':'Still in sparring posture; unsuitable normal window',
'63_1':'No unambiguous fight in selected window',
'66_1':'Standing interaction after incident; no clear fight',
'68_1':'Street gestures/possible shove ambiguous',
'69_0':'Distant group interaction; cannot confirm clean normal',
'69_1':'Distant small subjects; cannot confirm fight',
'71_0':'Assault occurs in proposed normal window',
'72_0':'Physical conflict begins inside proposed normal window',
'75_0':'Replay cut introduces conflict near end',
'76_0':'Distant action and camera movement; uncertain negative',
'77_0':'Guarded posture and opponent possibly off-screen',
'77_1':'Main interaction obscured by bench/framing',
'79_0':'Physical conflict begins near end of proposed normal window',
'81_0':'Participants already grappling at stair top',
'84_0':'Possible assault onset near end of proposed normal window',
'85_1':'Rotated and poorly framed; contact outside useful view',
'88_0':'Pushing begins near end of proposed normal window',
'92_0':'Contact begins near end of proposed normal window'}

def main():
    output=ROOT/'datasets/video_events/rebuild'
    data=json.loads((ROOT/'results/data_review/tnue_windows/candidates.json').read_text())
    assert set(REJECT)<=set(r['id'] for r in data['rows'])
    rows=[];review=[]
    for candidate in data['rows']:
        reason=REJECT.get(candidate['id'])
        review.append(dict(**candidate,decision='rejected' if reason else 'accepted_ai_provisional',
                           reason=reason or ('Visible interpersonal physical aggression' if candidate['label'] else 'No visible interpersonal physical aggression'),
                           reviewer='Codex visual review of temporal contact sheets',human_verified=False))
        if reason:continue
        row=dict(candidate);row.pop('status')
        # Recorded clips reuse actors/locations/multiviews: keep ALL in one group.
        row['group']='tnue-recorded-all' if 'recorded' in Path(row['path']).parts else 'tnue-'+Path(row['path']).stem
        row.update(annotation_source='AI temporal visual annotation; NOT official action labels',human_verified=False)
        rows.append(row)
    groups=sorted({r['group'] for r in rows if r['group']!='tnue-recorded-all'})
    random.Random(42).shuffle(groups)
    count=max(1,round(len(groups)*.15))
    assignment={g:'test' if i<count else 'validation' if i<2*count else 'train' for i,g in enumerate(groups)}
    assignment['tnue-recorded-all']='train'
    for row in rows:row['split']=assignment[row['group']]
    manifest=dict(dataset='TNUE-Fight Detection',labels=['normal','fight'],rows=rows,
                  status='provisional_ai_labels_pending_human_acceptance',
                  protocol='Fixed seed 42 source-group split: 15% groups test, 15% validation, remainder train; all recorded actors/scenes grouped in train; VFD-2000 external only',
                  annotation_protocol='141 candidate windows visually inspected in temporal contact sheets; ambiguous/mixed candidates rejected; timestamps refer only to reviewed windows, not whole-video labels',
                  limitation='Public subset and AI action labels; internal metrics are provisional and are NOT official TNUE benchmark scores',
                  public_videos=95,used_source_videos=len({r['path'] for r in rows}),
                  class_counts=dict(Counter(r['label'] for r in rows)),
                  split_counts={s:dict(Counter(r['label'] for r in rows if r['split']==s)) for s in ('train','validation','test')})
    for split,counts in manifest['split_counts'].items():
        assert set(counts)=={0,1},(split,counts)
    (output/'tnue_action_review.json').write_text(json.dumps(dict(rows=review,excluded_sources=data['excluded_sources']),indent=2))
    (output/'tnue_manifest.json').write_text(json.dumps(manifest,indent=2))
    print({k:v for k,v in manifest.items() if k not in ('rows','protocol','annotation_protocol')})

if __name__=='__main__':main()
