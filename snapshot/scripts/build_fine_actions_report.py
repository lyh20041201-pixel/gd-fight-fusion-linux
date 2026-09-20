"""Reproducible final Chinese report and local video comparison viewer."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import math
import numpy as np
from collections import Counter
from statistics import mean
from scripts.fine_actions_bc import ROOT,OUT,CACHE,read,write,sha,POLICY,CLASSES

ZH={'fall':'跌倒过程','rising':'起身','sitting_down':'坐下','bending':'弯腰',
    'lying_down':'主动躺下','seated':'坐姿','lying':'卧姿','other':'其他正常动作'}


def wilson(k,n):
    z=1.959964;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h,c+h]


def main():
    gmd=read(OUT/'gmd_test_summary.json');external=read(OUT/'external_fallvision_summary.json')
    annotations=read(OUT/'fine_annotations.json');rows=[r for r in annotations['videos'] if r['split']=='test']
    successes=[]
    for seed in POLICY['seeds']:
        b,c=gmd[f'B_{seed}'],gmd[f'C_{seed}']
        successes.append((c['fp']<=b['fp']-1 and c['tp']>=b['tp']) or (c['tp']>=b['tp']+2 and c['fp']<=b['fp']))
    promising=sum(successes)>=2
    decision='多分类达到本轮继续扩大试验的预设标准' if promising else '多分类未达到本轮继续扩大试验的预设标准'
    baseline=external['A']['recall']
    rejection={name: m['recall']<baseline-.02 for name,m in external.items() if name!='A'}
    stats={}
    for split in ['train','validation','test']:
        selected=[r for r in annotations['videos'] if r['split']==split]
        stats[split]=dict(videos=len(selected),seconds=sum(r['duration'] for r in selected),
            normal=sum(r['category']=='ADL' for r in selected),falls=sum(r['category']=='Fall' for r in selected),
            class_videos={c:sum(any(s['label']==c for s in r['spans']) for r in selected) for c in CLASSES})
    histories={f'{a}_{s}':read(OUT/a/f'seed_{s}'/'history.json') for a in POLICY['arms'] for s in POLICY['seeds']}
    total_seconds=sum(sum(e['seconds'] for e in h) for h in histories.values())
    storage_bytes=sum(p.stat().st_size for folder in [OUT,CACHE] for p in folder.rglob('*') if p.is_file())
    result=dict(decision=decision,paired_seeds_meeting_standard=sum(successes),promising_multiclass=promising,
        external_recall_rejection=rejection,live_model_changed=False,data=stats,
        training_epochs={k:len(h) for k,h in histories.items()},training_seconds=total_seconds,
        storage_bytes_before_report=storage_bytes,downloads_bytes=0,
        normal_test_seconds=gmd['A']['normal_video_seconds'],normal_test_ready_seconds=gmd['A']['normal_ready_seconds'])
    write(OUT/'conclusion.json',result)

    lines=['# 跌倒精细标签 B/C 实验结果','',f'**{decision}。本轮没有替换线上模型。**','',
      '本轮完成已有视频的 AI 时间段细化、RGB 复核、3 个随机种子的 B/C 配对训练，以及冻结阈值后的两套测试。用户无需标注。',
      '', '## 同一输入下的测试结果','',
      'A 为当前上线跌倒权重的模型基准；B 为精细时间标签二分类，C 为相同标签下八分类。A 与 B/C 的训练数据不同，因此 A→B 的差异不能单独归因于精细标注；B→C 才是受控的分类粒度对照。',
      '', '| 模型 | 检出跌倒 / 17 | 漏检 | 正常误报 / 20 | 检出事件中位延迟 | 正常片段报警簇 |',
      '|---|---:|---:|---:|---:|---:|']
    for name,m in gmd.items():
        delay='—' if m['median_detected_delay_seconds'] is None else f'{m["median_detected_delay_seconds"]:.2f} 秒'
        lines.append(f'| {name.replace("_"," / ")} | {m["tp"]} | {m["fn"]} | {m["fp"]} | {delay} | {m["normal_alerts"]} |')
    lines+=['',f'预设标准：三个配对种子中至少两个，C 比 B 少 ≥1 段正常误报且不损失召回，或多检出 ≥2 段跌倒且不增加正常误报。本次满足 {sum(successes)}/3。',
      '', '## 旧 FallVision 测试场景回归','',
      '138 段：69 跌倒、69 正常。复用已审计的旧骨架缓存及粗粒度视频标签，按视频取最高分；不改阈值。此结果不能替代连续视频的每小时误报或精确动作边界评价。',
      '', '| 模型 | 检出跌倒 / 69 | 漏检 | 正常误报 / 69 | 骨架完全缺失 |', '|---|---:|---:|---:|---:|']
    for name,m in external.items():lines.append(f'| {name.replace("_"," / ")} | {m["tp"]} | {m["fn"]} | {m["fp"]} | {m["unknown"]} |')
    lines+=['','若外部场景召回相对 A 降低超过 2 个百分点，按预设标准拒绝替换。各模型结果见 conclusion.json；本轮无论单一指标多好，都缺少有代表性的本地连续视频验收证据。',
      '', '## 误报率与统计边界','',
      f'GMD 正常测试总时长仅 **{gmd["A"]["normal_video_seconds"]:.1f} 秒**，其中已有足够历史且骨架可用的计分时长约 **{gmd["A"]["normal_ready_seconds"]:.1f} 秒**。这些是动作短视频，每段重新启动 4 秒缓存；不是数小时连续监控。',
      '', '| 模型 | 正常报警簇 / 总正常小时（外推） | 正常报警簇 / 可计分正常小时（外推） | 非正常片段内的提前、迟到或重复报警簇 |',
      '|---|---:|---:|---:|']
    for name,m in gmd.items():lines.append(f'| {name} | {m["false_alerts_per_normal_hour"]:.1f} | {m["false_alerts_per_ready_normal_hour"]:.1f} | {m["other_false_or_duplicate_alerts"]} |')
    lines+=['','每小时数字只展示这些短片的比率，不是长期误报率估计。零次误报也不能承诺实际使用零误报。17 段跌倒中 1 段差异就是 5.9 个百分点，而且测试人物只有 1 位，不能将窗口数当成独立样本数。',
      '', f'A/B/C 使用相同有效性规则。测试中 {gmd["A"]["all_unknown_videos"]} 段视频全程无法计分；含跌倒时仍算漏检。所有测试时点中 {gmd["A"]["unknown_window_fraction"]:.1%} 无法计分（主要是每段开头积累历史）。这不是模型主动拒识率。',
      '', '时间按源视频平均帧率重建。延迟是从来源跌倒起点到回放判断时点的差值，不包含摄像头传输、骨架提取和线上通知耗时；仅对成功检出的事件计算，不能替代端到端响应延迟。',
      '', '## 标注与模型具体做了什么','',
      '- 160 段十帧 RGB 时间轴全部进行 AI 检查，24 段进行更密集的边界复核。原始发布者的描述和时间戳原样保留；AI 精细边界另存。**没有独立人工复核，不声称逐帧精确真值。**',
      '- 动态跌倒之后是卧姿；正常躺下与跌倒的区别以来源描述和动作序列为依据，而不是单张卧姿截图。俯卧撑、深蹲归“其他正常动作”。',
      '- 每次只使用已到达的过去 4 秒、32 个真实采样时点，至少 8 个有效骨架时刻。过去窗口中仍有跌倒过程时，随后起身不会撤销已发生的跌倒。',
      '- C 的动作标签描述过去窗口内的最近过渡（跌倒优先），不是当前一帧的姿势；八分类混淆矩阵只对照 AI 派生标签，属于探索性分析。',
      '- B/C 相同 ST-GCN++ / NTU60 预训练主干，相同骨架、逐视频与二分类权重、批次顺序和训练预算；初始跌倒概率匹配。区别是二分类交叉熵与八分类交叉熵。',
      '- 单人数据选择历史有效骨架最多的轨迹；没有验证多人情况下的轨迹归属。A 也是在相同输入上的模型回放基准，不含摄像头采集、画质保护、起身过滤及线上事件网关。',
      '- 公开标签包含床上侧倒、后倒，也包含主动上床躺下；二者在 2D 骨架上可能很相近。所有数据均为演示动作，本轮没有真实意外跌倒的独立现场标签。',
      '- 80 段训练（人物 1、2）、43 段验证选模型和阈值（人物 3）、37 段本轮测试（人物 4）。全部六个模型选定后才计算测试预测。GMD、FallVision 都有此前实验历史，不称为新的盲测。',
      '', '## 资源与复现','',
      f'- 新下载：0 字节；复用原视频与模型权重。训练共 {sum(len(h) for h in histories.values())} 个 epoch，日志记录约 {total_seconds/60:.1f} 分钟；报告生成前新增实验目录和缓存合计约 {storage_bytes/1024**2:.1f} MiB。',
      '- 输入原视频未修改；线上 config/live_actions.json 及原训练结果未修改。',
      '- `protocol_frozen.json`、`fine_annotations.json`：预先封存的协议、标签、来源与代码哈希。',
      '- `B/seed_*/selection.json`、`C/seed_*/selection.json`：模型及阈值选择记录。',
      '- `evaluation/`：逐视频、逐窗口分数与混淆矩阵；`review/`：RGB 复核证据。',
      '- `tests/test_fine_actions_bc.py`：因果采样、无未来标签、分组隔离、初始概率匹配、A 基准移植一致性与未知／提前报警计数测试。',
      '', '复现入口（使用训练环境的 Python）：', '', '```powershell',
      'python scripts/prepare_fine_labels_bc.py poses', 'python scripts/train_fine_actions_bc.py all',
      'python scripts/evaluate_fine_actions_external.py evaluate', 'python scripts/build_fine_actions_report.py', '```',
      '', '已完成模型会校验哈希后跳过；模型协议和训练代码有变动时校验失败，需要新实验版本。',
    ]
    qa=read(OUT/'pose_input_audit.json')
    lines+=['','## 各动作的视频覆盖','',
        '一段视频可能包含多个动作，以下不是互斥计数。细动作范围由 AI 复核标签确定。',
        '', '| 动作 | 训练视频 | 验证视频 | 测试视频 |','|---|---:|---:|---:|']
    for c in CLASSES:lines.append(f'| {ZH[c]} | {stats["train"]["class_videos"][c]} | {stats["validation"]["class_videos"][c]} | {stats["test"]["class_videos"][c]} |')
    lines+=['','## 输入质量检查','',
        f'1480 个可计分窗口中，{qa["multiple_eligible_track_fragments"]} 个包含多段符合资格的轨迹片段，{qa["low_primary_coverage_windows"]} 个主轨迹有效帧覆盖不足 50%。轨迹片段多不等于画面里同时有多人；这里经常是同一个人在变姿势时被重新编号。',
        '', '抽查每个人物／类别中骨架覆盖最低的窗口，能看到侧躺、倒地出画、俯卧撑导致的人体关键点缺失。这说明骨架输入质量也需要改善，仅补充动作名称无法补回丢失的观测。诊断图：review/pose_quality.jpg；完整统计：pose_input_audit.json。',
    ]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    prediction_data={}
    for name in gmd:
        prediction_data[name]=read(OUT/'evaluation'/f'{name}_gmd_test.json')['predictions']
    clips=[]
    for r in rows:
        action_names=[ZH[c] for c in CLASSES[1:5] if any(s['label']==c for s in r['spans'])]
        if r['category']=='Fall':display_label='跌倒'
        elif 'push-up' in r['annotation'].lower():display_label='正常 · 俯卧撑'
        else:display_label='正常 · '+('、'.join(action_names) if action_names else '日常活动')
        clips.append(dict(id=r['id'],truth='跌倒' if r['category']=='Fall' else '正常',duration=r['duration'],
            display_label=display_label,
            description=r['annotation'].split(',')[4].strip(),spans=r['spans'],events=r['events'],
            predictions={name:[dict(t=round(p['end'],3),score=None if p['score'] is None else round(p['score'],5),
                 action=CLASSES[int(np.argmax(p['probabilities']))] if name.startswith('C') and p['probabilities'] else None,
                 reason=p['reason']) for p in predictions if p['video_id']==r['id']] for name,predictions in prediction_data.items()}))
    payload=dict(gmd=gmd,external=external,clips=clips,conclusion=result,zh=ZH)
    template=(ROOT/'scripts/fine_actions_report_template.html').read_text(encoding='utf-8')
    safe=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    (OUT/'index.html').write_text(template.replace('__PAYLOAD__',safe),encoding='utf-8')
    write(OUT/'progress.json',dict(status='complete',decision=decision,live_model_changed=False))
    print(decision,flush=True)


if __name__=='__main__':
    main()
