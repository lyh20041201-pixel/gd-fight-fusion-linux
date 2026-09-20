"""Build standalone experiment acceptance records; leave reports and PPT alone."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/video_events/rebuild_acceptance'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def link(path,label):return f'[{label}](<{Path(path).resolve().as_posix()}>)'

def main():
    OUT.mkdir(exist_ok=True)
    tasks=[('跌倒','gmdcsa24_r3d18_rebuild','FallVision','fallvision_external_final'),
           ('打架','tnue_r3d18_rebuild','VFD-2000','vfd_external_final')]
    summary=[];models=[]
    for name,directory,external,external_directory in tasks:
        base=ROOT/'results/video_events'/directory;run=read(base/'run_record.json');selection=read(base/'selection.json')
        evaluated=read(base/external_directory/'evaluation.json')
        if run['status']!='complete' or evaluated['status']!='complete':raise ValueError('Incomplete experiment')
        checkpoint=base/'selected_best.pt';sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if sha!=evaluated['checkpoint_sha256'] or sha!=selection['checkpoint_sha256']:raise ValueError('Checkpoint identity mismatch')
        chosen=run['stages'][selection['selected_stage']]
        row=dict(task=name,dataset=run['dataset'],internal_test=chosen['threshold_test'],external_dataset=external,
                 external_test=evaluated['metrics'],external_samples=evaluated['expected_samples'],
                 checkpoint=str(checkpoint),checkpoint_sha256=sha,threshold=chosen['threshold'],
                 selected_stage=selection['selected_stage'],annotation_status=run.get('annotation_status','audited_source_annotations'),
                 run_record=str(base/'run_record.json'),evaluation=str(base/external_directory/'evaluation.json'))
        summary.append(row);models.append((base,run,evaluated))
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        for stage,info in run['stages'].items():
            history=info['history'];epochs=[r['epoch'] for r in history]
            axes[0].plot(epochs,[r['loss'] for r in history],label=stage)
            axes[1].plot(epochs,[r['macro_f1'] for r in history],label=stage)
        axes[0].set(xlabel='Epoch',ylabel='Training loss');axes[1].set(xlabel='Epoch',ylabel='Validation macro F1',ylim=(0,1))
        for ax in axes:ax.grid(alpha=.25);ax.legend()
        fig.suptitle(run['dataset']+' | R3D-18 | seed 42');fig.savefig(OUT/f'{directory}_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,item,(_,run,_) in zip(axes,summary,models):
        cm=np.array(item['external_test']['confusion_matrix']);ax.imshow(cm,cmap='Blues')
        for (i,j),value in np.ndenumerate(cm):ax.text(j,i,str(value),ha='center',va='center',color='white' if value>cm.max()*.6 else 'black',fontsize=14)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=run['labels'],yticklabels=run['labels'],xlabel='Prediction',ylabel='Source label',title=item['external_dataset'])
    fig.savefig(OUT/'external_confusion_matrices.png',dpi=170);plt.close(fig)
    status=dict(training_and_external_evaluation='complete',models=summary,no_further_downloads=True,
                human_acceptance='pending',tnue_action_labels='AI provisional; not official action annotations',
                old_airtlab_files='Deletion blocked by automatic approval review; files remain on disk; default old checkpoint reference removed',
                runtime='New checkpoints saved for acceptance; default event checkpoint list remains empty',
                reports_and_ppt='Not changed in this rebuild')
    (OUT/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    md=['# 重训与跨数据集测试验收记录','','两套重训和两套跨数据集测试已完成。新权重已保存，尚待人工验收。报告和 PPT 未修改。','',
        '打架模型在 VFD-2000 上正常视频误报较多；本轮结果需要如实验收，不能仅用内部测试成绩证明泛化效果。',
        'TNUE 的行为标签是 AI 初标，内部成绩为暂定结果；其官方公开 CSV 标注的是人员框，未被当作打架真值。','',
        '| 模型 | 内部测试 Macro-F1 | 外部数据集 | 外部视频数 | 外部准确率 | 外部 Macro-F1 |',
        '|---|---:|---|---:|---:|---:|']
    for item in summary:
        md.append(f"| {item['task']} | {item['internal_test']['macro_f1']:.4f} | {item['external_dataset']} | {item['external_samples']} | {item['external_test']['accuracy']:.2%} | {item['external_test']['macro_f1']:.4f} |")
    md+=['','## 数据清单','',
         '| 用途 | 数据 | 本地审计后实际使用 |','|---|---|---|',
         '| 跌倒训练与内部测试 | GMDCSA-24 | 160 个视频、161 个标注片段；受试者 1+2 训练、3 验证、4 测试 |',
         '| 打架训练与内部测试 | TNUE-Fight Detection | 公开下载 95 个视频，76 个源视频中保留 102 个 AI 标注片段；训练 75、验证 15、测试 12 |',
         '| 跌倒外部测试 | FallVision | 5866 个原始 RGB 视频，按文件哈希去重后 5642 个；未使用遮罩视频和关键点副本 |',
         '| 打架外部测试 | VFD-2000 | 官方清单 2370 条，去掉 14 个重复项、2 组冲突标签内容，再排除与 TNUE 共用事件的 11 段，最终 2343 段 |','',
         'FallVision 有 7 个文件的头部帧数比实际多 1～2 帧，按可解码帧数校正时长后独立补测；原文件和原标签未改。修正前结果及补测结果均保留。',
         'TNUE 的 video418.mp4 重新下载后仍为相同哈希，无法解码，未使用。video490.mp4 与 video438.mp4 画面重复，已排除前者。',
         'VFD 与 TNUE 的交叉检查包括完全相同文件及采样画面感知哈希复核。确认共享的酒店、加油站事件按整个源 URL 排除；仍不能保证检出所有重新剪辑的同源画面。','',
         '## 训练和测试口径','',
         'R3D-18 使用 Kinetics-400 通用预训练权重初始化，重新训练分类头并进行末级微调，没有加载旧 AIRTLab 任务权重。每个输入为 16 帧、112×112 letterbox、4 秒以内视频窗口。',
         '两套模型均仅按内部验证集选择基线版本和阈值。外部测试扫描整段视频，4 秒窗口、2 秒步长，加上末尾窗口，取最高正类分数；未使用外部标签调参。内部与外部评价单位和窗口聚合方式有差异。',
         'GMDCSA-24 正类包含源数据集的床上跌倒和倒地状态；VFD 正类包含拳击等竞技画面，验收时需注意与实际监控场景的差别。','',
         '## 权重、视频、截图与逐条预测','']
    for item,(base,run,evaluated) in zip(summary,models):
        md += [f"### {item['task']} · {run['dataset']}",'',
               f"选中 {item['selected_stage']}，阈值 {item['threshold']:.2f}。",'',
               '- '+link(item['checkpoint'],'选中权重'),'- '+link(item['run_record'],'完整训练日志与超参数'),
               '- '+link(base/'acceptance/README.md','内部测试原视频、输入帧与预测'),
               '- '+link(item['evaluation'],'外部逐视频预测、所有窗口分数和混淆矩阵'),
               '- '+link(OUT/f'{base.name}_curves.png','训练曲线'),'']
        md+=['外部误报和漏报示例：','']
        for truth,pred,title in [(0,1,'误报'),(1,0,'漏报')]:
            example=next((r for r in evaluated['predictions'] if r['label']==truth and r['prediction']==pred and r.get('evidence')),None)
            if example:
                md += [f"- {title}：{link(example['path'],'原视频')} · {link(example['evidence'],'触发窗口截图')} · 正类分数 {example['probabilities'][1]:.4f}"]
        md+=['']
    md += [link(OUT/'external_confusion_matrices.png','两套外部测试混淆矩阵'),'',
           '## 清理与验收状态','',
           '旧 AIRTLab 默认权重引用已移除；新模型等待验收，默认事件模型列表为空。',
           '旧数据及旧模型的删除被自动审批审查拒绝（blocked by policy），文件仍在磁盘；待删除清单：'+link(ROOT/'datasets/video_events/rebuild/old_data_retirement_inventory.json','清理清单')+'。',
           '已按用户要求停止后续下载，余下工作仅使用本地文件。',
           '验证：相关视觉与配置测试 22 项通过；内部划分的源组和文件哈希无交集；外部清单均为独立测试用途；顺序解码与旧跳帧解码在 3 段对照视频上得到完全相同的输入张量。','']
    (OUT/'README.md').write_text('\n'.join(md),encoding='utf-8');print(str(OUT/'README.md'))

if __name__=='__main__':main()
