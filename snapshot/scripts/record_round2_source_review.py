"""Persist the bounded AI visual source audit; no inferred participant identities.

The observations below were checked on generated local contact sheets. They are
evidence against archive/filename splitting, NOT an exhaustive scene annotation.
"""
from collections import Counter, defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.skeleton_common import read,write,sha,digest,offline,CACHE,ROOT
from scripts.audit_skeleton_round2 import ROUND2,MANIFESTS,youtube_id
from scripts.skeleton_round2 import POLICY,require_source_seal


class Components:
    def __init__(self,values):self.parents={x:x for x in values}
    def find(self,x):
        while self.parents[x]!=x:x=self.parents[x]
        return x
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parents[max(a,b)]=min(a,b)


def main():
    offline()
    folder=ROUND2/'source_review'
    index=read(folder/'fallvision/index.json')
    # (sheet, one-based row) pairs; use actual IDs/paths from the persisted index.
    observed_links=[
        (0,1,11,1,'Same curtain, wooden chair and floral floor mat in positive/normal C_M_01'),
        (0,1,1,2,'Potential same floral-mat room and beige patterned curtain across M/N prefixes; conservatively merge'),
        (1,4,2,5,'Same floral-mat room, hanging clothes and ceiling/wall arrangement across N/D prefixes'),
        (2,1,3,8,'Same room with brown patterned curtains, door, pink storage unit and green grid mat'),
        (3,4,13,5,'Same red bedspread, gray panel and wooden dresser in fall/normal source groups'),
        (4,1,14,1,'Same S_M_01 room composition in positive/normal sources'),
        (5,1,15,1,'Same green wall, wooden dresser and doorway in S_N_01 positive/normal sources'),
        (5,12,7,10,'Same adjoining wooden doors, ceiling beam and blue floor cover across standing/bed categories'),
        (1,1,12,1,'Same chairs, green room and window arrangement in C_N_01 positive/normal sources'),
        (2,1,10,1,'Same C_D_0001 room, camera view and furniture in positive/normal sources'),
        (1,4,5,4,'Same floral-mat room and hanging clothes across chair/standing categories'),
        (0,1,4,1,'Potential same floral-mat room across chair/standing camera views; conservatively merge'),
        (7,4,6,1,'Matching black/green grid surface, floral cover and surrounding furniture across two bed archives'),
        (6,1,8,4,'Matching grid/floral covers and room furniture across N/M bed prefixes'),
        (8,4,9,5,'Matching grid/floral covers and room furniture across M/D bed prefixes'),
        (2,1,9,1,'Same brown-curtain room with door, pink storage and green grid mat across chair/bed categories')]
    edges=[];components=Components(x['group'] for x in index)
    for ai,ar,bi,br,note in observed_links:
        a,b=index[ai],index[bi]
        edges.append(dict(group_a=a['group'],group_b=b['group'],
            evidence_a=dict(sheet=a['sheet'],**a['selected'][ar-1]),
            evidence_b=dict(sheet=b['sheet'],**b['selected'][br-1]),
            finding=note,review_type='AI_visual_source_review',human_confirmed=False,
            decision='conservative_archive_merge_pending_per_clip_scene_mapping'))
        components.join(a['group'],b['group'])
    mapping={x['group']:'fallvision:unresolved-component:'+components.find(x['group']) for x in index}
    fall=read(MANIFESTS/'fallvision_source_review_draft.json')
    reviewed={r['sample_id'] for x in index for r in x['selected']}
    anomalies=[]
    for row in fall['rows']:
        row.update(proposed_source_group=mapping[row['original_group']],new_split='quarantine_source_unresolved',
            source_review_status='AI_sampled_archive_review; per_clip_scene_mapping_unresolved',
            source_sheet_reviewed=row['sample_id'] in reviewed,
            source_evidence='source_review/fallvision_source_group_evidence.json',
            exclusion_reason='No reliable per-clip filming-source map; conservative archive-component cannot be divided across splits')
        if row['original_group']=='nf_raw_s_1' and Path(row['path']).name.startswith('B_'):
            anomalies.append(dict(sample_id=row['sample_id'],path=row['path'],original_label=row['original_label'],
                original_origin=row['origin'],reason='B-prefixed file in standing archive; origin metadata requires scene/action review; normal label not changed'))
    fall.update(training_allowed=False,source_group_count=len(set(mapping.values())),
        source_review_limitation='192 stratified clips inspected; not all 4686 clips assigned to independently verified filming sessions/scenes. M/N/D are not established participant IDs.',
        retained_source_pool_count=len(fall['rows']),eligible_sealed_training_rows=0)
    write(MANIFESTS/'fallvision_source_review_draft.json',fall)
    fall_evidence=dict(status='blocked_for_sealing',reviewed_contact_sheets=len(index),
        visually_sampled_clips=len(reviewed),source_pool_clips=len(fall['rows']),
        per_clip_verified_scene_map_complete=False,person_independent_claim=False,
        human_confirmed=False,edges=edges,archive_to_conservative_component=mapping,
        conservative_components=len(set(mapping.values())),
        feasibility='One coarse conservative component cannot populate three source-disjoint splits with both labels. This is not a claim that all clips are one actual session.',
        missing_evidence='An exhaustive clip-to-filming-scene/session mapping with cross-category and cross-prefix equivalences, or reliable local recording metadata.',
        label_anomalies=anomalies,label_changes=0,
        limitation='A different finer grouping may be possible after further source work; current sampled evidence does not validate one.')
    write(folder/'fallvision_source_group_evidence.json',fall_evidence)

    candidate=read(folder/'vfd_cross_url_candidates.json')
    vfd=read(MANIFESTS/'vfd_source_review_draft.json');by={r['sample_id']:r for r in vfd['rows']}
    components=Components(youtube_id(r['original_group']) for r in vfd['rows'])
    decisions=[]
    for i,c in enumerate(candidate['candidates']):
        item=dict(c,candidate_index=i,evidence_sheet=str(folder/'vfd_candidate_sheets'/f'{i//12:02d}.jpg'),sheet_row=i%12+1)
        # Match left/right source IDs to actual images, not sorted pair labels.
        item['left_source_id']=youtube_id(by[c['a']['sample_id']]['original_group'])
        item['right_source_id']=youtube_id(by[c['b']['sample_id']]['original_group'])
        item.update(review_status='AI_visual_candidate_pair_reviewed',human_confirmed=False,
            same_event=True if i==0 else None,
            decision='merge_sources_same_edited_sequence' if i==0 else 'conservatively_merge_unresolved_source_relation',
            observation='Same outdoor action-to-interview image sequence; different source URLs and SHA-256; both source labels are positive' if i==0 else
                'Representative cached strips inspected. This does not establish that the complete two source URLs have no shared event; compilations and transformed footage require full event mapping.',
            label_action='retain_original_labels')
        decisions.append(item);components.join(c['source_a'],c['source_b'])
    groups=defaultdict(list)
    for row in vfd['rows']:groups[components.find(youtube_id(row['original_group']))].append(row)
    for key,rows in groups.items():
        gid='vfd:conservative-component:'+key
        for row in rows:
            row.update(proposed_source_group=gid,new_split='unassigned',
                source_review_status='canonical_url_and_conservative_candidate_merge; full_event_map_unverified',
                source_evidence='source_review/vfd_source_group_evidence.json',exclusion_reason=None)
    vfd.update(source_group_count=len(groups),training_allowed=False,
        source_review_limitation='Canonical URLs and 129 reviewed representative pHash pairs; not exhaustive temporal mapping of all events in compilations.',
        retained_source_pool_count=len(vfd['rows']),eligible_sealed_training_rows=0)
    write(MANIFESTS/'vfd_source_review_draft.json',vfd)
    evidence=dict(status='conservative_group_draft_not_sealed',canonical_source_ids=168,
        reviewed_candidate_pairs=len(decisions),confirmed_same_sequence_pairs=1,
        unresolved_source_level_pairs=len(decisions)-1,decisions=decisions,
        conservative_components=len(groups),
        components=[dict(group='vfd:conservative-component:'+k,rows=len(rs),
            source_ids=sorted({youtube_id(r['original_group']) for r in rs}),labels=dict(Counter(r['label'] for r in rs))) for k,rs in sorted(groups.items(),key=lambda x:-len(x[1]))],
        split_feasibility='Not declared impossible for VFD. Conservative URL groups are only a draft; no retained test was sealed.',
        limitation='No original labels changed. Similarity alone is not proof of duplication; unresolved relationships are merged conservatively, not declared distinct.',
        prior_exclusion_sources=[dict(path=str(p),sha256=sha(p)) for p in [ROOT/'datasets/video_events/rebuild/vfd_source_exclusions.json',ROOT/'datasets/video_events/rebuild/vfd_duplicate_audit.json']],
        human_confirmed=False)
    write(folder/'vfd_source_group_evidence.json',evidence)
    write(ROUND2/'planned_protocol.json',POLICY)
    preflight=[]
    for name in ('fallvision','vfd'):
        p=MANIFESTS/f'{name}_source_review_draft.json'
        try:require_source_seal(p)
        except ValueError as exc:preflight.append(dict(dataset=name,training_allowed=False,reason=str(exc)))
        else:raise AssertionError('A source draft unexpectedly passed the training gate')
    write(ROUND2/'source_gate.json',dict(status='NOT_PASSED',training_started=False,
        sealed_splits=0,datasets=preflight,models_completed=0,models_planned=12,seeds=[42,43,44],epochs_completed=0,
        reason='FallVision reliable three-way filming-source split not established; VFD canonical/merged groups remain provisional',
        source_bytes_and_cache_verified=7029,source_pool_count=7029,labels_modified=0,
        excluded_original_bed_positive_clips_preserved=956,training_inference_parity='not_run_no_round2_models',resume_training_parity='not_run_no_round2_training',
        old_model_recalibration='not_run_no_sealed_validation',retained_test_evaluation='not_run_no_sealed_test',
        regression_evaluation='not_run_no_round2_models; GMD and TNUE retain first-round training history'))
    write(ROUND2/'progress.json',dict(stage='blocked_before_split_sealing',file_cache_audit=dict(completed=7029,total=7029),
        fallvision_sampled_source_review=dict(sheets_completed=16,sheets_total=16,clips_reviewed=192,source_clips_total=4686),
        vfd_candidate_pair_review=dict(completed=129,total=129),models=dict(completed=0,total=12),
        current_model=None,seed=None,epoch=0,training_started=False,
        training_eta=None,training_eta_reason='No authorized leak-free training split and no measured training throughput'))
    print('SOURCE REVIEW SAVED; no split sealed, no model training started.',flush=True)


if __name__=='__main__':main()
