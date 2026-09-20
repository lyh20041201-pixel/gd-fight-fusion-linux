"""Second-round results and local evidence; no modifications to reports or slides."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import html,os
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from scripts.skeleton_common import CACHE,ROOT,read,sha,offline
from scripts.skeleton_io import write
from scripts.skeleton_round2 import OUT,MANIFESTS,require_source_seal
from scripts.build_skeleton_acceptance import render

LABELS={'round1_original_threshold':'第一轮原阈值','round1_validation_recalibrated':'第一轮新验证集校准','round2':'第二轮重新训练'}

def figures(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    legends=['Round1 original threshold','Round1 recalibrated on new validation','Round2 retrained']
    colors=['#9aa4af','#e2a84b','#287dac']
    for row,name in enumerate(['fallvision','vfd']):
        summary=read(OUT/'evaluations'/name/'retained_test/summary.json')
        for col,attr in enumerate(['recall','normal_false_positive_rate']):
            ax=axes[row,col]
            for j,(variant,legend) in enumerate(zip(LABELS,legends)):
                values=[summary['variants'][variant]['aggregate'][mod][attr] for mod in ['rgb','skeleton']]
                means=[100*(v['mean'][1] if attr=='recall' else v['mean']) for v in values]
                stds=[100*(v['std'][1] if attr=='recall' else v['std']) for v in values]
                ax.bar(np.arange(2)+(j-1)*.24,means,.24,yerr=stds,capsize=3,label=legend,color=colors[j])
            ax.set_xticks([0,1],['RGB','Skeleton']);ax.set_ylabel('Percent; mean +/- sample SD');ax.set_ylim(0,105)
            ax.set_title(name+' retained test: '+('event recall' if col==0 else 'normal false-positive proportion'))
            if col:ax.axhline(5,color='#b93b39',linestyle='--',linewidth=1)
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0,0].legend(fontsize=8);fig.savefig(folder/'retained_test_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for row,name in enumerate(['fallvision','vfd']):
        for col,mod in enumerate(['rgb','skeleton']):
            ax=axes[row,col]
            for seed in [42,43,44]:
                base=OUT/name/mod/f'seed_{seed}';record=read(base/'run_record.json');selected=read(base/'selection.json')
                history=[r for rows in record['histories'].values() for r in rows]
                x=[r['cumulative_epoch'] for r in history];y=[100*r['validation']['recall'][1] for r in history]
                line,=ax.plot(x,y,label=f'seed {seed}',marker='.',markersize=3)
                ax.scatter([selected['cumulative_epoch']],[100*selected['validation']['recall'][1]],marker='*',s=90,color=line.get_color())
            ax.set_title(name+' / '+mod+' validation');ax.set_xlabel('Cumulative epoch (all stages)');ax.set_ylabel('Event recall at normal FPR <= 5%')
            ax.set_ylim(0,105);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(folder/'validation_learning_curves.png',dpi=180);plt.close(fig)

def fmt(aggregate,metric,event=False):
    data=aggregate[metric];mean=data['mean'][1] if event else data['mean'];std=data['std'][1] if event else data['std']
    return f'{mean*100:.2f} ± {std*100:.2f}'

def build():
    offline();torch.set_num_threads(2);folder=OUT/'acceptance_round2';folder.mkdir(parents=True,exist_ok=True)
    from scripts.audit_round2_model_selection import verify
    selection_audit=verify()
    lines=['# 第二轮离线动作识别实验验收','',
        '两条路线分别重新训练，YOLO-Pose 权重、骨架提取、4 秒/32 时刻/2 秒步长和补尾窗保持第一轮协议。RGB 从本地 Kinetics R3D-18 初始化；骨架动作模型随机初始化。种子为 42、43、44。第一轮文件保持原样。','',
        '以下均为重新划分后的保留测试，FallVision 和 VFD 在第一轮已有评测历史，不是全新盲测。模型、轮次、阶段和阈值均仅按新验证集选择；正常误报比例约束为不超过 5%，再优先事件召回率。测试结果未用于回调阈值。','',
        '表中为百分数，三种子均值 ± 样本标准差（ddof=1）。unknown 保留在总数及对应真值类别召回率分母，正样本 unknown 计漏检。']
    condition=['# 条件分层与残留问题','', '人物大小、可见度和多人条件是固定骨架缓存推导的观测代理；不等于人工遮挡标签。原始来源的 Chair/Stand 标记与真实起始姿态也并不完全等价。']
    media=[];verifications={}
    for name,title in [('fallvision','跌倒 / FallVision'),('vfd','打架 / VFD-2000')]:
        manifest=require_source_seal(MANIFESTS/f'{name}.json');dest=OUT/'evaluations'/name
        summary=read(dest/'retained_test/summary.json');reg=read(dest/'regression/summary.json')
        assert summary['same_denominators_all_models']
        models=[read(OUT/name/mod/f'seed_{s}'/'selection.json') for mod in ['skeleton','rgb'] for s in [42,43,44]]
        assert all(x['status']=='complete' and x['test_accessed'] is False and x['epochs_trained']<=50 for x in models)
        lines+=['',f'## {title}','',f'保留 {len(manifest["rows"])} 段；训练/验证/测试：'+ ' / '.join(str(sum(r['split']==s for r in manifest['rows'])) for s in ['train','validation','test'])+'。',
            '',manifest['limitation'],'', '| 模型 | 参考方式 | 事件精确率 | 事件召回率 | 宏 F1 | 正常误报比例 | 覆盖率 |','|---|---|---:|---:|---:|---:|---:|']
        deltas=[]
        for mod in ['rgb','skeleton']:
            for variant,values in summary['variants'].items():
                a=values['aggregate'][mod]
                lines.append(f'| {mod} | {LABELS[variant]} | {fmt(a,"precision",True)} | {fmt(a,"recall",True)} | {fmt(a,"macro_f1")} | {fmt(a,"normal_false_positive_rate")} | {fmt(a,"coverage")} |')
            old=summary['variants']['round1_validation_recalibrated']['aggregate'][mod]
            new=summary['variants']['round2']['aggregate'][mod]
            delta=100*(new['recall']['mean'][1]-old['recall']['mean'][1]);fpr=new['normal_false_positive_rate']['mean']
            deltas+=['',f'{mod} 相对“第一轮固定权重＋同准则验证集校准”的事件召回率变化为 {delta:+.2f} 个百分点；第二轮测试正常误报比例均值 {fpr:.2%}。这组比较用于区分重训与阈值变化，不能据此作因果或显著性保证。']
        lines+=deltas
        lines+=['','逐种子模型选择与测试约束：','', '| 路线 | 种子 | 实际训练总轮次 | 保留阶段 / 累计轮次 | 验证有效检测 | 测试误报≤5% |','|---|---:|---:|---|---|---|']
        for mod in ['rgb','skeleton']:
            for seed in [42,43,44]:
                s=read(OUT/name/mod/f'seed_{seed}'/'selection.json');ok=summary['test_fpr_target_met']['round2'][f'{mod}_{seed}']
                lines.append(f'| {mod} | {seed} | {s["epochs_trained"]} | {s["selected_stage"]} / {s["cumulative_epoch"]} | {"有" if s["validation"]["effective_detection"] else "未获得有效检测能力"} | {"是" if ok else "否，保持验证阈值如实报告"} |')
        lines+=['',f'完整逐种子指标、含 unknown 的混淆矩阵、均值/标准差及分层数据：[保留测试结果](evaluations/{name}/retained_test/summary.json)。',
            f'逐片段分数：[索引](evaluations/{name}/retained_test/prediction_index.json)；[误报、漏报与无法判断](evaluations/{name}/retained_test/error_unknown_index.json)。',
            '',f'### {reg["dataset"].upper()} 能力回归（原划分）','',reg['description'],
            '', '| 原划分 | 样本数 | 路线 | 第一轮原阈值召回 | 第一轮校准召回 | 第二轮召回 | 第二轮误报比例 |','|---|---:|---|---:|---:|---:|---:|']
        for split,rs in reg['by_original_split'].items():
            for mod in ['rgb','skeleton']:
                a={v:rs['variants'][v]['aggregate'][mod] for v in LABELS}
                lines.append(f'| {split} | {rs["samples"]} | {mod} | {fmt(a["round1_original_threshold"],"recall",True)} | {fmt(a["round1_validation_recalibrated"],"recall",True)} | {fmt(a["round2"],"recall",True)} | {fmt(a["round2"],"normal_false_positive_rate")} |')
        lines+=['',f'回归完整指标见 [原划分回归结果](evaluations/{name}/regression/summary.json)。训练历史与 TNUE 暂定标签限制仍然有效。']
        condition+=['',f'## {title}','', '| 条件 | 子组 | 正常/事件 | RGB 召回均值 | 骨架召回均值 | 骨架正常误报均值 | 骨架覆盖均值 |','|---|---|---:|---:|---:|---:|---:|']
        strata=summary['variants']['round2']['strata']
        for attr,groups in strata.items():
            for val,metrics in groups.items():
                count=np.array(metrics['skeleton_42']['confusion_matrix']).sum(1)
                sk=[metrics[f'skeleton_{s}'] for s in [42,43,44]];rgb=[metrics[f'rgb_{s}'] for s in [42,43,44]]
                event=lambda data: f'{np.mean([x["recall"][1] for x in data]):.2%}' if count[1] else '无事件样本'
                fpr=f'{np.mean([x["normal_false_positive_rate"] for x in sk]):.2%}' if count[0] else '无正常样本'
                condition.append(f'| {attr} | {val} | {count[0]} / {count[1]} | {event(rgb)} | {event(sk)} | {fpr} | {np.mean([x["coverage"] for x in sk]):.2%} |')
        rows={r['sample_id']:r for r in manifest['rows']};records=[read(p) for p in sorted((dest/'retained_test/samples').glob('*.json'))]
        picks={}
        for category in ['false_positive','false_negative','positive_unknown','correct_event','correct_normal']:
            for r in records:
                p=r['comparisons']['round2']['skeleton_42']['prediction'];t=r['label']
                hit={'false_positive':t==0 and p==1,'false_negative':t==1 and p==0,'positive_unknown':t==1 and p==-1,'correct_event':t==1 and p==1,'correct_normal':t==0 and p==0}[category]
                if hit:picks.setdefault(r['sample_id'],(category,r));break
        for sid,(category,r) in picks.items():
            row=rows[sid];raw=torch.load(CACHE/name/(sid+'.pt'),map_location='cpu',weights_only=True)
            scores={m:r['comparisons']['round2'][f'{m}_42'] for m in ['rgb','skeleton']}
            peak=scores['skeleton']['peak_window'];peak=0 if peak is None else peak
            entry=render(row,raw,folder/'media'/name,scores=scores,window_index=peak)
            media.append(dict(dataset=name,category=category,sample_id=sid,label=row['label'],scores=scores,**entry))
        verifications[name]=dict(split_counts=manifest['split_counts'],source_seal_sha256=sha(MANIFESTS/f'{name}.seal.json'),
            inference_parity=read(dest/'validation_inference_parity.json'),same_test_denominators=summary['same_denominators_all_models'])
    lines+=['','## 解释边界','',
        '- 满足验证集误报约束不保证测试也满足；上表逐种子如实报告，不用测试结果再次选阈值。零召回即未获得有效检测能力。',
        '- 本轮没有改进姿态定位或骨架提取输入。骨架输入不足的正样本仍然漏检；训练收益只能来自动作分类器与验证集阈值选择。',
        '- 第二轮使用与其训练一致的 CPU 骨架预处理和 FP16 最终打分；第一轮保留原有数值路径。第一轮打架最终分数为 FP32，这一数值差异见 [打分精度记录](NUMERICAL_NOTES.md)，不能声称两轮数值路径完全相同。',
        '- FallVision 来源隔离带来数量和比例变化，隔离记录保留于新清单。没有可靠人员编号，不能声称人员独立。',
        '- 离线片段正常误报比例不是连续摄像头每小时误报次数；没有进行摄像头测试，不能承诺教室实测效果。',
        '- 本文件属于独立实验验收材料，没有修改项目报告或 PPT。','',
        '[原视频与骨架叠加验收页](acceptance_round2/index.html)；[条件分层分析](acceptance_round2/conditions.md)。']
    figures(folder)
    lines+=['','![统一保留测试比较](acceptance_round2/retained_test_comparison.png)','',
        '上图误差棒为三种子标准差；正常误报图中红虚线为 5%。','',
        '![验证集训练曲线](acceptance_round2/validation_learning_curves.png)','',
        '训练曲线仅来自验证集，星号标示最终保留轮次，不含测试集选模。']
    (OUT/'ROUND2_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (folder/'conditions.md').write_text('\n'.join(condition)+'\n',encoding='utf8')
    write(folder/'media_index.json',dict(selection='Fixed seed42, first sample_id in each listed outcome category; examples selected only after test inference, never for training decisions.',media=media))
    def url(path):return quote(Path(os.path.relpath(path,folder)).as_posix(),safe='/')
    cards=[]
    for r in media:
        cards.append(f'<article><h2>{r["dataset"]} · {r["category"]} · 源标签 {r["label"]}</h2><p>{r["sample_id"]}，固定种子 42；左原画面，右已有观测骨架。</p><video controls preload="none" src="{url(r["video"])}"></video><p><a href="{url(r["source"])}">完整原视频</a> · <a href="{url(r["poster"])}">采样序列图</a></p></article>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第二轮实验验收</title><style>body{max-width:1150px;margin:30px auto;padding:0 18px;font:16px system-ui;background:#f3f5f8;color:#192333}article{background:white;padding:20px;margin:24px 0}video,img{width:100%}p{line-height:1.6}</style><h1>第二轮 RGB 与骨架动作识别验收</h1><p>训练和保留测试评测已完成。这里是 AI 生成的验收材料，尚无人工确认。样例不是完整指标；全部 unknown 和错误均见索引。</p><p><a href="../ROUND2_RESULTS.md">完整三种子比较</a> · <a href="conditions.md">条件分层</a></p><img src="retained_test_comparison.png" alt="三种子统一保留测试比较"><details><summary>验证集训练曲线</summary><img src="validation_learning_curves.png"></details>'+''.join(cards)+'</html>'
    (folder/'index.html').write_text(page,encoding='utf8')
    before=read(OUT/'audit/first_round_inventory.json')['files']
    def check(r):
        p=Path(r['path']);return None if p.exists() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'] else r['path']
    with ThreadPoolExecutor(max_workers=3) as pool:changed=[x for x in pool.map(check,before) if x]
    if changed:raise ValueError('First-round files changed: '+str(changed))
    caches_changed=[]
    for name in ['fallvision','vfd']:
        draft=read(MANIFESTS/f'{name}_source_review_draft.json')
        for r in draft['rows']:
            a=r['audit_record'];st=Path(a['cache_path']).stat()
            if st.st_size!=a['freshness']['cache_size'] or st.st_mtime_ns!=a['freshness']['cache_mtime_ns']:caches_changed.append(a['cache_path'])
    if caches_changed:raise ValueError('Original sample caches changed')
    write(OUT/'verification/round2_final.json',dict(status='complete_offline_experiment',datasets=verifications,models_complete=12,
        first_round_files_rehashed=len(before),first_round_files_changed=changed,original_caches_checked=7029,original_caches_changed=caches_changed,
        training_checks=read(OUT/'verification/training_checks.json'),media_videos=len(media),media_frames=sum(r['frames'] for r in media),
        evaluator_integration=read(OUT/'verification/evaluator_integration.json'),source_input_eligibility=read(OUT/'verification/skeleton_eligibility_by_split.json'),
        model_selection_audit=selection_audit,
        report_or_ppt_modified=False,camera_tested=False,human_acceptance=False))
    print('ROUND2 ACCEPTANCE COMPLETE',len(media),'videos; first round unchanged',len(before),flush=True)

if __name__=='__main__':build()
