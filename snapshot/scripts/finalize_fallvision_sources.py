"""Seal audited scene containment; no participant IDs or model-driven relabeling."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict,Counter
from scripts.skeleton_common import CACHE,read,sha,seal,offline
from scripts.skeleton_round2 import OUT,MANIFESTS
from scripts.seal_round2_sources import publish

def main():
    offline();folder=OUT/'source_review/fallvision_scenes_v2'
    working=read(folder/'source_edges_working.json');layout=read(folder/'clusters.json');review=read(folder/'ai_review_working.json')
    assert len(working['rows'])==4686 and review['reviewed_clusters']==list(range(96))
    assert working['review_sha256']==sha(folder/'ai_review_working.json')
    by={r['sample_id']:dict(r) for r in working['rows']};names=defaultdict(list)
    # Same white-wall scene family as C_M_110. A different view of the paired-door
    # room cannot confidently be excluded; quarantine the whole visible family.
    for r in by.values():
        if (r['cluster']==4 and r['position']<24) or (r['cluster']==82 and r['position'] in [30,43]):
            r['scene']='quarantine';r['resolution_reason']='plain-wall/blue-curtain scene could not be reliably separated from paired-door room; entire visually matched family isolated'
        names[r['canonical_basename']].append(r)
    for members in names.values():
        if any(r['scene']=='quarantine' for r in members):
            for r in members:r['scene']='quarantine'
        assert len({r['scene'] for r in members})==1
    candidates=[]
    ambiguous_paired_indices={46,90,208,242,290,293,419,420}
    for i,p in enumerate(working['cross_group_candidates']):
        a,b=by[p['a']],by[p['b']]
        if a['scene']=='quarantine' or b['scene']=='quarantine':
            decision='quarantined_source_family';basis='At least one complete canonical-name family isolated before splitting.'
        elif 'computer_lab' in [a['scene'],b['scene']]:
            decision='AI_scene_difference';basis='Exhaustive source-sheet review shows rows of desktop monitors and office desks versus a domestic bed/sofa/cabinet scene. Matching coarse padded-frame pHash does not establish a shared source. Same participants are not ruled out.'
        else:
            assert i not in ambiguous_paired_indices
            decision='AI_scene_difference';basis='All 54 paired-door candidate strips inspected; this pair has different fixed room anchors (ornate paired doors/ceiling beam versus bed/cabinet/floral-mat/TV room). Plain-wall candidates were isolated with their scene family.'
        candidates.append(dict(index=i,**p,decision=decision,evidence=basis,
            source_sheet_a=a['evidence'],source_sheet_b=b['evidence'],pair_strip=str(folder/f'cross_scene_{i//12:03d}.jpg')))
    evidence_path=folder/'source_review_sealed.json'
    source=read(CACHE/'manifests/fallvision.json');retained=[];isolated=[];assignments=[]
    for original in source['rows']:
        r=dict(original);a=by[r['sample_id']]
        r.update(original_label=r['label'],original_group=r['group'],original_split=r['split'],
            original_source_label=r.get('source_label'),group='fallvision_scene_'+a['scene'],
            source_review_status='conservative_merged' if a['scene']=='other_domestic_conservative' else 'AI_reviewed',
            human_confirmed=False,canonical_source_basename=a['canonical_basename'],
            source_evidence=str(evidence_path),source_contact_sheet=a['evidence'],source_contact_position=a['position'],
            source_audit_record=str(OUT/'audit/samples/fallvision'/(r['sample_id']+'.json')),
            label_review_status='source label retained; inherited scope exclusions and AI source review, not individual human confirmation')
        assignments.append(dict(sample_id=r['sample_id'],original_group=r['original_group'],new_source_group=r['group'],
            canonical_basename=a['canonical_basename'],contact_sheet=a['evidence'],position=a['position'],review_status=r['source_review_status']))
        if a['scene']=='quarantine':
            r.update(split='quarantine',new_source_group=r['group'],exclusion_reason=a.get('resolution_reason','Scene anchors too limited to rule out shared source; ambiguity propagated to every canonical-name variant.'),source_review_status='AI_source_ambiguous')
            isolated.append(r)
        else:retained.append(r)
    evidence=dict(status='complete_for_scene_containment',reviewer='AI',human_confirmed=False,participant_independent=False,
        initial_population=4686,retained=len(retained),newly_quarantined=len(isolated),source_labels_changed=0,
        counts={g:dict(Counter(r['label'] for r in retained if r['group']==g)) for g in sorted({r['group'] for r in retained})},
        visual_review=dict(tiles=4686,layout_clusters=96,contact_pages=sum(len(c['pages']) for c in layout['clusters']),
            image_method='Median of first/middle/last already cached RGB timestamps, solely for source review; representative three-frame strips for cross-scene ambiguity. No new pose extraction.',
            extra_pair_review='All 54 paired-door cross-scene strips inspected; 515 lab-versus-domestic candidates resolved using the exhaustive source-sheet scene mapping, with additional closest-pair strip checks. Not all 515 pair strips separately inspected.',
            preliminary_archive_review='Earlier archive-only coarse merging was superseded by exhaustive per-clip scene evidence; archive numbers were never treated as people.'),
        scene_definitions=dict(paired_doors='Visually linked ornate wooden doors, ceiling beam, wall fittings and blue floral surface; include reverse camera views with beige door/window and chair variants.',
            computer_lab='Rows of desktop monitors on long wooden desks, office chairs and cream curtains; all camera directions and source-name variants merged.',
            other_domestic_conservative='Many distinct domestic rooms conservatively kept in one indivisible group. This group is not one claimed session or person; incomplete relationships inside it cannot cross splits.'),
        assignments=assignments,quarantine=isolated,candidate_dispositions=candidates,
        canonical_filename_cross_group_relations=0,uncontained_documented_relations=0,
        limitations='Scene containment based on local directory names and AI visual anchors, with no verified participant/session metadata. Only three coarse containment groups, including one large conservative group; holdout scenes are narrow and ratios deviate substantially. No participant independence claim; undetected source transformations remain possible.',
        source_files_sha256={str(p):sha(p) for p in [folder/'clusters.json',folder/'ai_review_working.json',folder/'source_edges_working.json']},
        contact_sheet_sha256={p['path']:sha(p['path']) for c in layout['clusters'] for p in c['pages']})
    seal(evidence_path,evidence)
    publish('fallvision',retained,dict(inherited_scope_exclusions=source['exclusions'],round2_source_ambiguity=isolated),
        [str(evidence_path)],evidence['limitations'])
    print('FALLVISION RETAINED',len(retained),'ISOLATED',len(isolated),flush=True)

if __name__=='__main__':main()
