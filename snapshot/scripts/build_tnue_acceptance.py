"""Readable audit trail for provisional TNUE labels and internal predictions."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def link(path,label):
    return f'[{label}](<{Path(path).resolve().as_posix()}>)'

def main():
    out=ROOT/'results/video_events/tnue_r3d18_rebuild';dest=out/'acceptance';dest.mkdir(exist_ok=True)
    data=json.loads((out/'manifest.json').read_text());record=json.loads((out/'run_record.json').read_text())
    selected=json.loads((out/'selection.json').read_text())['selected_stage']
    predictions=json.loads((out/f'{selected}_test_predictions.json').read_text())
    labels=['正常','打架']
    md=['# TNUE 重训验收材料','','状态：AI 初标，待人工验收。不是官方 TNUE 行为标注，也不是官方基准成绩。',
        '公开下载得到 95 段视频；其中 1 段源文件损坏。本轮从 76 个源视频选出 102 个行为片段，未标记的时间段不参与训练。',
        '人员框 CSV 未用于推断行为标签；141 个候选窗口经过时序截图复核，剔除 39 个混合或不明确的片段。',
        'R3D-18 以 Kinetics-400 通用预训练权重初始化，重新训练分类头和末级微调，没有加载旧 AIRTLab 权重。',
        '按源视频分组、固定 seed 42 划分；所有自录视频因演员、地点和多视角重用，统一放在训练组。',
        '75 段训练（正常 25、打架 50），15 段验证（正常 7、打架 8），12 段测试（各 6）。',
        '验证集选择 baseline，固定阈值 0.60。外部 VFD-2000 未用于训练、挑选权重或调阈值。','',
        '| 版本 | 验证 Macro-F1 | 内部测试 Macro-F1 | 误报 | 漏报 |','|---|---:|---:|---:|---:|']
    for stage,info in record['stages'].items():
        test=info['threshold_test'];cm=test['confusion_matrix']
        md.append(f"| {stage} | {info['best_validation_macro_f1']:.4f} | {test['macro_f1']:.4f} | {cm[0][1]}/6 | {cm[1][0]}/6 |")
    md+=['','## 暂定内部测试逐条结果','','| 片段 | AI 真值 | 预测 | 打架分数 | 原视频 | 输入帧 |','|---|---|---|---:|---|---|']
    for pred in predictions:
        row=next(r for r in data['rows'] if r['path']==pred['path'] and r['start']==pred['start'])
        md.append(f"| {row['id']} · {row['start']}-{row['end']:.2f}s | {labels[row['label']]} | {labels[int(pred['candidate'])]} | {pred['probabilities'][1]:.4f} | {link(row['path'],'视频')} | {link(row['evidence'],'16 帧')} |")
    md+=['','## 全部纳入片段','','| ID | 划分 | AI 标签 | 时间窗 | 原视频 | 截图 |','|---|---|---|---|---|---|']
    for row in data['rows']:
        md.append(f"| {row['id']} | {row['split']} | {labels[row['label']]} | {row['start']}-{row['end']:.2f}s | {link(row['path'],'视频')} | {link(row['evidence'],'16 帧')} |")
    md+=['',link(ROOT/'datasets/video_events/rebuild/tnue_action_review.json','全部候选的纳入/排除理由'),'','报告和 PPT 尚未修改。']
    (dest/'README.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(str(dest/'README.md'))

if __name__=='__main__':main()
