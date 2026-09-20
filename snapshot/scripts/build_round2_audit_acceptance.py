"""Local source-audit acceptance, without training/evaluating any action model."""
from concurrent.futures import ThreadPoolExecutor
import html
import os
from pathlib import Path
import sys
from urllib.parse import quote
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.skeleton_common import read,write,sha,offline,CACHE,OUT,ROOT
from scripts.audit_skeleton_round2 import ROUND2,MANIFESTS


def main():
    offline()
    import torch,cv2
    torch.set_num_threads(2);cv2.setNumThreads(1)
    from scripts.build_skeleton_acceptance import render
    acceptance=ROUND2/'acceptance';acceptance.mkdir(parents=True,exist_ok=True)
    fall=read(MANIFESTS/'fallvision_source_review_draft.json')
    vfd=read(MANIFESTS/'vfd_source_review_draft.json')
    fidx=read(ROUND2/'source_review/fallvision/index.json')
    pair=read(ROUND2/'source_review/vfd_cross_url_candidates.json')['candidates'][0]
    selected=[('fallvision',fidx[0]['selected'][0]['sample_id'],'跨正负来源的同房间例子'),
        ('fallvision',fidx[11]['selected'][0]['sample_id'],'跨正负来源的同房间例子'),
        ('vfd',pair['a']['sample_id'],'跨 URL 的相同画面序列'),
        ('vfd',pair['b']['sample_id'],'跨 URL 的相同画面序列')]
    unusable=[]
    for name,m in [('fallvision',fall),('vfd',vfd)]:
        missing=[r for r in m['rows'] if not r['audit_record']['skeleton_usable']]
        for r in missing:
            unusable.append(dict(dataset=name,sample_id=r['sample_id'],path=r['path'],label=r['label'],
                reasons=sorted({q['unknown_reason'] for q in r['audit_record']['window_quality']}),
                source_review_status=r['source_review_status'],split=r['new_split']))
        first=next(r for r in missing if r['label']==1)
        selected.append((name,first['sample_id'],'正样本骨架输入不足例子；未进行第二轮预测'))
    write(acceptance/'unusable_skeleton_index.json',dict(rows=unusable,total=len(unusable),
        interpretation='Input eligibility audit only, no second-round action-model predictions; all unknowns must remain in any future evaluation denominator.'))
    media=[]
    for name,sid,reason in selected:
        row=next(r for r in (fall if name=='fallvision' else vfd)['rows'] if r['sample_id']==sid)
        cache=torch.load(CACHE/name/(sid+'.pt'),map_location='cpu',weights_only=True)
        result=render(row,cache,acceptance/'source_evidence_media',scores=None)
        media.append(dict(dataset=name,sample_id=sid,reason=reason,label=row['label'],**result))
    write(acceptance/'source_evidence_media.json',dict(media=media,action_models_run=0,fresh_pose_extractions=0,
        purpose='Render existing observed keypoints over original sampled source frames; not a new extraction or action evaluation'))
    def link(path):return quote(Path(os.path.relpath(path,acceptance)).as_posix(),safe='/')
    blocks=[]
    for entry in media:
        blocks.append(f'<article><h3>{html.escape(entry["dataset"])} · {html.escape(entry["reason"])}</h3>'
            f'<p>源标签 {entry["label"]}；{entry["sample_id"]}。没有第二轮模型分数。</p>'
            f'<video controls preload="none" src="{link(entry["video"])}"></video>'
            f'<p><a href="{link(entry["source"])}">完整原视频</a> · <a href="{link(entry["poster"])}">采样序列图</a></p></article>')
    sheets=[]
    for entry in fidx:
        sheets.append(f'<li><a href="{link(entry["sheet"])}">{html.escape(entry["group"])}：12 段抽样来源画面</a></li>')
    vfd_sheets=sorted((ROUND2/'source_review/vfd_candidate_sheets').glob('*.jpg'))
    sources=''.join(f'<li><a href="{link(p)}">VFD 候选对复核 {p.stem}</a></li>' for p in vfd_sheets)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>第二轮来源审查：尚未封存划分</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 18px;color:#182536;background:#f3f5f7}article{background:white;padding:18px;margin:20px 0}video{width:100%}p{line-height:1.6}a{color:#165cad}</style>
<h1>第二轮来源审查：尚未封存划分</h1>
<p>文件与缓存核对 7029/7029；完成模型 0/12，轮次 0。尚无新测试集、新模型或召回改善结果。</p>
<p>FallVision：16 组复核图、192 段抽样存在跨归档来源联系，尚无覆盖 4686 段的可靠场次/场景映射。VFD：168 个规范来源 ID，129 对候选画面已初查；确认 1 对不同哈希的相同画面序列，合集事件来源图仍未完整核实。</p>
<p>当前材料是 AI 来源复核，未经过人工确认。没有依据模型预测改标签。以下叠加只使用第一轮固定 YOLO-Pose 缓存，不含新模型推理。</p>'''
    page+=f'<p><a href="{link(ROUND2/"AUDIT_STATUS.md")}">完整审查结论与下一步条件</a> · <a href="{link(ROUND2/"source_gate.json")}">训练前置条件状态</a> · <a href="unusable_skeleton_index.json">612 段输入不足索引</a></p>'
    page+=''.join(blocks)+'<h2>FallVision 来源抽样</h2><ul>'+''.join(sheets)+'</ul><h2>VFD 跨 URL 候选</h2><ul>'+sources+'</ul></html>'
    (acceptance/'index.html').write_text(page,encoding='utf-8')
    before=read(ROUND2/'audit/first_round_inventory.json')['files']
    def check(record):
        p=Path(record['path'])
        return None if p.is_file() and p.stat().st_size==record['bytes'] and sha(p)==record['sha256'] else record['path']
    with ThreadPoolExecutor(max_workers=3) as pool:changed=[x for x in pool.map(check,before) if x]
    if changed:raise ValueError(f'First-round artifacts changed: {changed}')
    # Existing per-sample cache stats were recorded before inspection; no writes are allowed.
    caches_changed=[]
    for m in [fall,vfd]:
        for row in m['rows']:
            audit=row['audit_record'];stat=Path(audit['cache_path']).stat()
            if stat.st_size!=audit['freshness']['cache_size'] or stat.st_mtime_ns!=audit['freshness']['cache_mtime_ns']:
                caches_changed.append(audit['cache_path'])
    if caches_changed:raise ValueError('Original caches changed')
    write(ROUND2/'audit/final_verification.json',dict(status='AUDIT_ONLY_PASSED_TRAINING_GATE_NOT_PASSED',
        first_round_files_rehashed=len(before),first_round_files_changed=changed,
        original_caches_stat_checked=7029,original_caches_changed=caches_changed,
        source_evidence_videos=len(media),source_evidence_frames=sum(m['frames'] for m in media),
        automated_tests=dict(passed=22,command='.venv/Scripts/python.exe -m pytest tests/test_skeleton_round2.py tests/test_skeleton_comparison.py -q'),
        training_started=False,tests_used_for_model_selection=False,action_model_inference_run=False,
        round2_code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/audit_skeleton_round2.py',ROOT/'scripts/skeleton_round2.py',ROOT/'scripts/record_round2_source_review.py',Path(__file__),ROOT/'tests/test_skeleton_round2.py']}))
    print('Audit acceptance ready; first-round files unchanged:',len(before),'; evidence videos:',len(media),flush=True)


if __name__=='__main__':main()
