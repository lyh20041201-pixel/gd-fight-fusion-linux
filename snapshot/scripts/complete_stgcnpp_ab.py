"""Build A/B comparison, source-frame evidence, and immutable-input acceptance."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import html,os
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
from scripts.stgcnpp_ab import *
from scripts.build_skeleton_acceptance import render

def fmt(a,key,event=False):
    mean=a[key]['mean'];std=a[key]['std']
    if event:mean,std=mean[1],std[1]
    return f'{100*mean:.2f} ± {100*std:.2f}'

def figures(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for i,name in enumerate(POLICY['datasets']):
        summary=read(OUT/'evaluations'/name/'retained_test/summary.json')
        for j,attr in enumerate(['recall','normal_false_positive_rate']):
            ax=axes[i,j];top=0
            for k,arm in enumerate(['A','B']):
                a=summary['aggregate'][arm][attr];mean=a['mean'][1] if attr=='recall' else a['mean'];std=a['std'][1] if attr=='recall' else a['std']
                ax.bar(k,100*mean,yerr=100*std,capsize=4,color=['#8597ad','#16798b'][k])
                ax.text(k,100*mean+100*std+2,f'{100*mean:.1f}%',ha='center')
                top=max(top,100*(mean+std)+8)
            ax.set_xticks([0,1],['A: random','B: NTU60 pretrained']);ax.set_ylim(0,max(105 if j==0 else 12,top))
            ax.set_title(name+' / '+('event recall' if j==0 else 'normal false-positive proportion'))
            ax.set_ylabel('Percent; mean ± sample SD');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            if j:ax.axhline(5,ls='--',color='#bc3c3c',label='5% target');ax.legend()
    fig.savefig(folder/'ab_retained_test.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for i,name in enumerate(POLICY['datasets']):
        for j,arm in enumerate(['A','B']):
            ax=axes[i,j]
            for seed in POLICY['seeds']:
                dest=OUT/name/arm/f'seed_{seed}';h=read(dest/'run_record.json')['history'];s=read(dest/'selection.json')
                line,=ax.plot([r['epoch'] for r in h],[100*r['validation']['recall'][1] for r in h],label=f'seed {seed}')
                ax.scatter([s['epoch']],[100*s['validation']['recall'][1]],marker='*',s=90,color=line.get_color())
            ax.set_title(name+' / '+arm);ax.set_xlabel('Epoch');ax.set_ylabel('Validation recall at normal FPR ≤ 5%');ax.set_ylim(0,105);ax.grid(alpha=.2);ax.legend()
    fig.savefig(folder/'ab_validation_curves.png',dpi=160);plt.close(fig)

def main():
    setup_runtime();assert read(OUT/'evaluations/complete.json')['status']=='complete'
    folder=OUT/'acceptance';folder.mkdir(parents=True,exist_ok=True)
    lines=['# ST-GCN++ 随机初始化与 NTU60 预训练初始化对照','',
        'A 组随机初始化主干；B 组加载 NTU60 HRNet 二维 Joint 预训练主干。两组使用同一种 ST-GCN++、相同的任务头初值、样本、划分、动作窗口、数据顺序、优化器和训练预算。每个任务分别训练种子 42、43、44。',
        '', '所有参数从第一轮参与专项训练；每模型最多 50 轮、早停耐心值 8。模型轮次和阈值只由验证集选择：正常片段误报比例≤5% 时优先事件召回，再较低误报、较高宏 F1。B 包含先前通用预训练计算，因此本实验控制专项训练预算，不是全生命周期计算量相同。',
        '', '表格为百分数，均值 ± 三种子样本标准差。unknown 保留在分母，正样本 unknown 计为漏检。当前测试有此前评测历史，属于固定保留测试/能力回归，不是新盲测。']
    tables=[];media=[];condition=['# 条件分层','', '大小、可见度和多人条件是已有姿态缓存推导的代理指标；来源 Chair/Stand 不等于逐片段人工确认。']
    for name,title in [('fallvision','跌倒 / FallVision'),('vfd','打架 / VFD-2000')]:
        dest=OUT/'evaluations'/name;summary=read(dest/'retained_test/summary.json');manifest=require_source_seal(MANIFESTS/f'{name}.json')
        lines+=['',f'## {title}','',str(manifest['limitation']),'',f'保留测试 {summary["samples"]} 段。',
            '', '| 组别 | 事件精确率 | 事件召回率 | 宏 F1 | 正常误报比例 | 覆盖率 |','|---|---:|---:|---:|---:|---:|']
        for arm in ['A','B']:
            a=summary['aggregate'][arm];cells=[arm,fmt(a,'precision',True),fmt(a,'recall',True),fmt(a,'macro_f1'),fmt(a,'normal_false_positive_rate'),fmt(a,'coverage')]
            lines.append('| '+' | '.join(cells)+' |');tables.append([title,*cells])
        delta=summary['paired_B_minus_A']['recall']['mean'][1]*100;fprdelta=summary['paired_B_minus_A']['normal_false_positive_rate']['mean']*100
        lines+=['',f'B−A 的平均事件召回率变化为 {delta:+.2f} 个百分点，正常误报比例变化为 {fprdelta:+.2f} 个百分点。此为固定数据与训练方案下的描述性结果，不宣称普遍收益或统计显著性。',
            '', '| 组别/种子 | 训练轮数 | 保留轮次 | 验证阈值 | 事件精确率 | 事件召回率 | 宏 F1 | 正常误报比例 | 覆盖率 | 验证有效检测 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for arm in ['A','B']:
            for seed in POLICY['seeds']:
                s=read(OUT/name/arm/f'seed_{seed}'/'selection.json');m=summary['per_seed'][f'{arm}_{seed}']
                lines.append(f'| {arm}/{seed} | {s["epochs_trained"]} | {s["epoch"]} | {s["threshold"]:.9g} | {m["precision"][1]:.2%} | {m["recall"][1]:.2%} | {m["macro_f1"]:.2%} | {m["normal_false_positive_rate"]:.2%} | {m["coverage"]:.2%} | '+('有' if s['validation']['effective_detection'] else '未获得有效检测能力')+' |')
        failures=[k for k,ok in summary['test_fpr_target_met'].items() if not ok]
        lines+=['', '测试正常误报超过 5% 的模型：'+(', '.join(failures) if failures else '无')+'。保持验证集阈值，不依据测试回调。',
            '',f'[完整指标与含 unknown 混淆矩阵](evaluations/{name}/retained_test/summary.json) · [逐片段分数](evaluations/{name}/retained_test/prediction_index.json) · [误报/漏报/无法判断](evaluations/{name}/retained_test/error_unknown_index.json)。',
            '', '混淆矩阵行是真实正常/事件，列是预测正常/事件/unknown。']
        for key,m in summary['per_seed'].items():lines.append(f'- {key}: `{m["confusion_matrix"]}`')
        old=read(ROOT/'results/video_events/skeleton_comparison_round2/evaluations'/name/'retained_test/summary.json')['variants']['round2']['aggregate']['skeleton']
        lines+=['',f'第二轮骨架参考：召回率 {fmt(old,"recall",True)}，正常误报比例 {fmt(old,"normal_false_positive_rate")}。其网络、输入特征和分数精度与本实验不同；不能把它与 B 的差异全归于预训练。',
            '', '原划分能力回归（GMD/TNUE 有第一轮训练历史；TNUE 标签仍暂定；本次未参与训练或选择）：','',
            '| 原划分 | 数量 | 组别 | 事件精确率 | 事件召回率 | 宏 F1 | 正常误报比例 | 覆盖率 |','|---|---:|---|---:|---:|---:|---:|---:|']
        for split in ['train','validation','test']:
            path=dest/'regression'/split/'summary.json'
            if not path.exists():continue
            reg=read(path)
            for arm in ['A','B']:
                a=reg['aggregate'][arm]
                lines.append(f'| {split} | {reg["samples"]} | {arm} | {fmt(a,"precision",True)} | {fmt(a,"recall",True)} | {fmt(a,"macro_f1")} | {fmt(a,"normal_false_positive_rate")} | {fmt(a,"coverage")} |')
        condition+=['',f'## {title}','','| 条件 | 子组 | 正常/事件 | A 召回 | B 召回 | A 误报 | B 误报 | 覆盖率 |','|---|---|---:|---:|---:|---:|---:|---:|']
        for attr,groups in summary['strata'].items():
            for val,g in groups.items():
                counts=np.asarray(g['per_seed']['A_42']['confusion_matrix']).sum(1)
                means={a:{k:float(np.mean([g['per_seed'][f'{a}_{s}'][k][1] if k=='recall' else g['per_seed'][f'{a}_{s}'][k] for s in POLICY['seeds']])) for k in ['recall','normal_false_positive_rate','coverage']} for a in ['A','B']}
                recall=lambda a:f'{means[a]["recall"]:.2%}' if counts[1] else '无事件样本'
                fpr=lambda a:f'{means[a]["normal_false_positive_rate"]:.2%}' if counts[0] else '无正常样本'
                condition.append(f'| {attr} | {val} | {counts[0]}/{counts[1]} | {recall("A")} | {recall("B")} | {fpr("A")} | {fpr("B")} | {means["A"]["coverage"]:.2%} |')
        rows={r['sample_id']:r for r in manifest['rows']};records=[read(p) for p in sorted((dest/'retained_test/samples').glob('*.json'))];picks={}
        for category in ['B_correct_A_miss','B_miss_A_correct','false_positive','positive_unknown','correct_event','correct_normal']:
            for r in records:
                a,b=r['results']['A_42']['prediction'],r['results']['B_42']['prediction'];t=r['label']
                hit={'B_correct_A_miss':t==1 and a!=1 and b==1,'B_miss_A_correct':t==1 and a==1 and b!=1,
                     'false_positive':t==0 and 1 in [a,b],'positive_unknown':t==1 and b==-1,'correct_event':t==1 and b==1,'correct_normal':t==0 and b==0}[category]
                if hit:picks.setdefault(r['sample_id'],(category,r));break
        for sid,(category,r) in picks.items():
            rawpath=CACHE/name/(sid+'.pt')
            if not rawpath.exists():continue
            raw=torch.load(rawpath,map_location='cpu',weights_only=True)
            if not raw['clips']:continue
            scores={k:r['results'][k] for k in ['A_42','B_42']};peak=scores['B_42']['peak_window']
            entry=render(rows[sid],raw,folder/'media'/name,scores=scores,window_index=0 if peak is None else peak)
            media.append(dict(dataset=name,sample_id=sid,category=category,label=r['label'],**entry))
    lines+=['','## 解释与验收','',
        '- 所有 12 次模型选择先封存，再访问保留测试；完整验证集重新推理与训练保存分数逐项相同。测试/回归结果不用于选择模型、轮次或阈值。',
        '- YOLO 权重、采样、匿名轨迹及缺失判断保持原规则；本次没有改善姿态提取覆盖率。新输入只是已有缓存的格式转换，没有重复提取骨架。',
        '- 主干是上游 ST-GCN++ 原计算代码，框架注册和 BN/ReLU 工厂以等价本地 PyTorch 实现替代；没有安装额外框架依赖。全部 690 个主干张量严格加载；NTU60 分类头弃用。',
        '- 官方预训练使用 HRNet、100 时刻；本次使用 YOLO、32 时刻及原 0.3 缺失掩码。两组转换相同。这些域差异可能限制迁移收益。',
        '- 本次两组都使用 FP16 主干计算、FP32 sigmoid，区别于第二轮的 FP16 sigmoid；不能将对第二轮的差异解释为纯初始化收益。',
        '- 来源标签、AI 来源复核与人工确认保持区分；未新增人工标签确认。FallVision 场景分组不代表人员独立。',
        '- 正常片段误报比例不是每小时摄像头误报次数，没有摄像头实测，也不承诺教室效果。未修改原始数据、旧实验、正式报告或 PPT。',
        '', '[逐条件分析](acceptance/conditions.md) · [原画面与骨架叠加](acceptance/index.html)','',
        '![A/B 保留测试](acceptance/ab_retained_test.png)','', '![验证训练曲线](acceptance/ab_validation_curves.png)']
    figures(folder)
    (OUT/'AB_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (folder/'conditions.md').write_text('\n'.join(condition)+'\n',encoding='utf-8')
    write(folder/'media_index.json',dict(selection='After frozen evaluation only; seed42, first sample_id for each listed outcome; no training decisions',media=media))
    def link(path):return quote(Path(os.path.relpath(path,folder)).as_posix(),safe='/')
    table='<table><tr>'+''.join('<th>'+v+'</th>' for v in ['任务','组别','事件精确率','事件召回率','宏 F1','正常误报比例','覆盖率'])+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in tables)+'</table>'
    cards=''.join(f'<article><h2>{r["dataset"]} · {r["category"]}</h2><p>种子 42，源标签 {r["label"]}；左原画面，右已有观测骨架。视频级分数不用于指认每个人。</p><video controls preload="none" src="{link(r["video"])}"></video><p><a href="{link(r["source"])}">完整原视频</a> · <a href="{link(r["poster"])}">采样序列图</a></p></article>' for r in media)
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ST-GCN++ A/B 对照</title><style>body{max-width:1150px;margin:30px auto;padding:0 18px;font:16px system-ui;background:#f3f5f8;color:#192333}article{background:white;padding:20px;margin:24px 0}video,img{width:100%}p{line-height:1.6}table{width:100%;border-collapse:collapse}td,th{padding:10px;border:1px solid #cbd1db;text-align:right}</style><h1>ST-GCN++：随机初始化 A 与 NTU60 预训练 B</h1><p>百分数，均值 ± 三种子标准差。保留测试有先前评测历史；unknown 全部保留。样例由 AI 按固定规则生成，未人工验收。</p><p><a href="../AB_RESULTS.md">完整结果与原划分回归</a> · <a href="conditions.md">条件分层</a></p>'+table+'<img src="ab_retained_test.png"><details><summary>验证集训练曲线</summary><img src="ab_validation_curves.png"></details>'+cards+'</html>'
    (folder/'index.html').write_text(page,encoding='utf-8')
    protected=read(OUT/'audit/protected_outputs.json')['files']
    def check(r):
        p=Path(r['path']);return p.exists() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i,(record,ok) in enumerate(zip(protected,pool.map(check,protected))):
            if not ok:raise ValueError('Protected artifact changed: '+record['path'])
            if (i+1)%1000==0:print('VERIFY_PROTECTED',i+1,'/',len(protected),flush=True)
    checked=0
    for name in ['fallvision','vfd','gmd','tnue']:
        for r in read(OUT/'audit'/f'inputs_{name}.json')['rows']:
            if sha(r['path'])!=r['source_sha256']:raise ValueError('Original source changed')
            if r['raw_cache_sha256'] is not None and sha(r['raw_cache'])!=r['raw_cache_sha256']:raise ValueError('Original pose cache changed')
            if sha(r['prepared_cache'])!=r['prepared_cache_sha256']:raise ValueError('Prepared input changed')
            checked+=1
    final=dict(status='complete',models=12,protected_files_unchanged=len(protected),input_records_verified=checked,media_videos=len(media),media_frames=sum(r['frames'] for r in media),
        pretrained_sha256=sha(WEIGHTS),test_thresholds_never_retuned=True,report_ppt_modified=False,camera_tested=False,human_acceptance=False,
        comparisons={name:read(OUT/'evaluations'/name/'retained_test/summary.json')['paired_B_minus_A'] for name in POLICY['datasets']})
    write(OUT/'verification/final.json',final);print('ACCEPTANCE_COMPLETE',final['models'],final['media_videos'],flush=True)

if __name__=='__main__':main()
