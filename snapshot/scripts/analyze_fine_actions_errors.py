"""Post-selection error descriptions; never feeds labels or settings to training."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
from scripts.fine_actions_bc import OUT,read,write,POLICY,CLASSES


def main():
    data=read(OUT/'gmd_test_summary.json')
    truth={r['id']:r for r in read(OUT/'fine_annotations.json')['videos'] if r['split']=='test'}
    lookup={name:{r['id']:r for r in m['videos']} for name,m in data.items()}
    rows=[]
    for sid,r in truth.items():
        votes={a:sum(lookup[f'{a}_{s}'][sid]['detected'] for s in POLICY['seeds']) for a in ['B','C']}
        info=dict(id=sid,source_type=r['category'],source_description=r['annotation'],
            actions=sorted({s['label'] for s in r['spans']}),B_positive_seeds=votes['B'],C_positive_seeds=votes['C'],
            A_positive=lookup['A'][sid]['detected'],ready_windows=lookup['A'][sid]['usable_windows'])
        rows.append(info)
    groups={
        'normal_with_lying_down':[r for r in rows if r['source_type']=='ADL' and 'lying_down' in r['actions']],
        'normal_with_rising':[r for r in rows if r['source_type']=='ADL' and 'rising' in r['actions']],
        'normal_with_sitting_down':[r for r in rows if r['source_type']=='ADL' and 'sitting_down' in r['actions']],
        'normal_with_bending':[r for r in rows if r['source_type']=='ADL' and 'bending' in r['actions']],
        'normal_pushups':[r for r in rows if r['source_type']=='ADL' and 'push-up' in r['source_description'].lower()],
        'bed_related_falls':[r for r in rows if r['source_type']=='Fall' and 'bed' in r['source_description'].lower()],
        'other_falls':[r for r in rows if r['source_type']=='Fall' and 'bed' not in r['source_description'].lower()],
    }
    summary={}
    for group,rs in groups.items():
        summary[group]=dict(videos=len(rs),A_positive_videos=sum(r['A_positive'] for r in rs),
            B_positive_mean=sum(r['B_positive_seeds'] for r in rs)/3,
            C_positive_mean=sum(r['C_positive_seeds'] for r in rs)/3,ids=[r['id'] for r in rs])
    result=dict(groups=summary,videos=rows,note='Post-hoc descriptive groups may overlap; majority/mean across seeds is diagnostic, not a deployed ensemble or a new selection criterion.')
    write(OUT/'error_analysis.json',result)
    for g,m in summary.items():print(g,m,flush=True)


if __name__=='__main__':main()
