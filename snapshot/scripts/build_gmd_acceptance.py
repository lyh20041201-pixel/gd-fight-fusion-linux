"""Generate review evidence from saved predictions; never modifies thesis/PPT."""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'results/video_events/gmdcsa24_r3d18_rebuild'
    evidence = out / 'acceptance'
    evidence.mkdir(exist_ok=True)
    rows = json.loads((out / 'baseline_test_predictions.json').read_text())
    record = json.loads((out / 'run_record.json').read_text())
    md = ['# GMDCSA-24 跌倒模型验收材料', '',
          '本文件仅为训练验收记录，论文、报告和 PPT 尚未修改。', '',
          '模型：R3D-18，以 Kinetics-400 通用预训练权重初始化，重新训练分类头；未加载旧 AIRTLab 权重。',
          '划分：受试者 1、2 训练，受试者 3 验证，受试者 4 测试。以内部验证集选择 baseline_best.pt。',
          '输入：每段标签区间中央至多 4 秒，均匀采样 16 帧，112×112 letterbox。',
          '标签含源数据集的床上跌倒和倒地状态，不等价于仅检测跌倒瞬间。', '',
          '| 版本 | 验证 Macro-F1 | 内部测试 Macro-F1 | 正常误报 | 跌倒漏报 |',
          '|---|---:|---:|---:|---:|']
    for stage, data in record['stages'].items():
        test = data['threshold_test']; cm = test['confusion_matrix']
        md.append(f"| {stage} | {data['best_validation_macro_f1']:.4f} | {test['macro_f1']:.4f} | {cm[0][1]}/20 | {cm[1][0]}/17 |")
    md += ['', '外部 FallVision 测试以独立 evaluation.json 为准；内部测试成绩不能替代跨数据集成绩。', '',
           '## 受试者 4 全部测试样本', '',
           '图片为模型所取时间窗的 16 帧。预测概率为 fall 类的 softmax 分数，未进行概率校准。', '',
           '| 样本 | 真值 | 预测 | 跌倒分数 | 原视频 | 输入帧 |', '|---|---|---|---:|---|---|']
    for i, row in enumerate(rows):
        cap = cv2.VideoCapture(row['path']); fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = float(row['start']); end = min(float(row['end']), n / fps)
        start = max(start, (start + end) / 2 - 2); end = min(end, start + 4)
        ids = np.linspace(round(start*fps), max(round(start*fps), min(n-1,round(end*fps)-1)),16).astype(int)
        frames=[]
        for idx in ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx)); ok, image = cap.read()
            if not ok:
                raise ValueError(row['path'])
            image = cv2.resize(image,(240,135))
            cv2.putText(image,f'{idx/fps:.2f}s',(4,18),0,.45,(0,255,255),1)
            frames.append(image)
        cap.release()
        name = f"{i+1:02d}_truth{row['label']}_pred{int(row['candidate'])}.jpg"
        cv2.imwrite(str(evidence/name),np.vstack([np.hstack(frames[j:j+4]) for j in range(0,16,4)]))
        row['evidence'] = str(evidence/name)
        labels=['正常','跌倒']; source=Path(row['path']).as_posix()
        md.append(f"| {i+1} | {labels[row['label']]} | {labels[int(row['candidate'])]} | {row['probabilities'][1]:.4f} | [视频](<{source}>) | [16 帧](<{(evidence/name).as_posix()}>) |")
    (evidence/'predictions.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (evidence/'README.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('Wrote evidence for',len(rows),'test segments')


if __name__ == '__main__':
    main()
