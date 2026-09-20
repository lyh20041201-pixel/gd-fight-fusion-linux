"""Local preview videos, complete per-sample indexes and comparison figures."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,html,time,os
from urllib.parse import quote
import cv2,numpy as np,torch
from scripts.skeleton_common import *
from scripts.skeleton_io import write
from backend.vision.skeleton_actions import EDGES

def fit(im,w=640,h=360):
    ih,iw=im.shape[:2];s=min(w/iw,h/ih);small=cv2.resize(im,(max(1,round(iw*s)),max(1,round(ih*s))))
    canvas=np.zeros((h,w,3),np.uint8);x=(w-small.shape[1])//2;y=(h-small.shape[0])//2
    canvas[y:y+small.shape[0],x:x+small.shape[1]]=small;return canvas

def render(row,cache,folder,scores=None,window_index=0):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    clip=cache['clips'][window_index];name=row['sample_id']+f'_w{window_index}'
    video=folder/(name+'.webm');poster=folder/(name+'.jpg');sidecar=folder/(name+'.json')
    wanted=dict(source_sha256=row['sha256'],cache_signature=cache['signature'],window_index=window_index,
                scores=scores,render_version=1)
    if sidecar.exists():
        old=read(sidecar)
        if old['config']==wanted and Path(old['video']).exists():return old
    # VP8 is available in the bundled OpenCV build; no codec download is attempted.
    fps=32/max(.01,clip['end']-clip['start'])
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'VP80'),fps,(1280,416))
    if not writer.isOpened():
        video=folder/(name+'.avi');writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),fps,(1280,416))
    if not writer.isOpened():raise RuntimeError('No local video encoder available')
    cap=cv2.VideoCapture(row['path']);frames=[]
    try:
        for t,idx in enumerate(clip['frame_indices']):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx));ok,im=cap.read()
            if not ok:raise ValueError(f'Evidence decode failed at {idx}')
            overlay=im.copy();h,w=im.shape[:2];thickness=max(2,round(max(h,w)/500))
            for person,tid in enumerate(clip['track_ids']):
                k=clip['keypoints'][person,t].numpy();box=clip['boxes'][person,t].numpy()
                visible=k[:,2]>=.3
                if not visible.any():continue
                color=((tid*67)%190+60,(tid*107)%190+60,(tid*41)%190+60)
                for u,v in EDGES:
                    if visible[u] and visible[v]:cv2.line(overlay,tuple(k[u,:2].astype(int)),tuple(k[v,:2].astype(int)),color,thickness)
                for point in k[visible]:cv2.circle(overlay,tuple(point[:2].astype(int)),thickness+1,color,-1)
                cv2.putText(overlay,f'ID {tid}',(int(box[0]),max(20,int(box[1]))),0,max(.5,max(h,w)/1600),color,thickness)
            canvas=np.zeros((416,1280,3),np.uint8);canvas[56:,:640]=fit(im);canvas[56:,640:]=fit(overlay)
            title=f'{row["sample_id"]} | source t={clip["timestamps"][t]:.2f}s | label={row["label"]}'
            cv2.putText(canvas,title,(10,20),0,.5,(230,230,230),1)
            labels='RAW VIDEO                                           OBSERVED KEYPOINTS + ANONYMOUS TRACKS'
            if scores:
                labels=' | '.join(f'{key}: '+('unknown' if val['score'] is None else f'{val["prediction"]} ({val["score"]:.3f})') for key,val in scores.items())
            cv2.putText(canvas,labels,(10,44),0,.5,(90,220,255),1);writer.write(canvas)
            if t in [0,8,16,24,31]:frames.append(canvas)
    finally:cap.release();writer.release()
    check=cv2.VideoCapture(str(video));count=int(check.get(cv2.CAP_PROP_FRAME_COUNT));ok,_=check.read();check.release()
    if not ok or count!=32:raise ValueError(f'Evidence video invalid: {video}; frames={count}')
    cv2.imwrite(str(poster),np.vstack([cv2.resize(f,(960,312)) for f in frames]))
    result=dict(config=wanted,video=str(video),poster=str(poster),source=row['path'],frames=count,
                description='32 sampled source frames with observed skeletons; scene-level predictions do not label every person as an aggressor')
    write(sidecar,result);return result

def make_status():
    selections=list(OUT.glob('*/*/seed_*/selection.json'))
    status=dict(updated_at=time.time(),models_complete=len(selections),models_expected=12,training='complete' if len(selections)==12 else 'in_progress')
    status['extraction']={n:read(OUT/f'extraction_{n}.json') if (OUT/f'extraction_{n}.json').exists() else {'status':'not_started'} for n in ['gmd','tnue','fallvision','vfd']}
    status['evaluation']={n:read(OUT/'evaluations'/n/'progress.json') if (OUT/'evaluations'/n/'progress.json').exists() else {'status':'not_started'} for n in ['gmd','tnue']}
    status['overall']='awaiting_acceptance' if all((OUT/'evaluations'/n/'summary.json').exists() for n in ['gmd','tnue']) and (OUT/'acceptance/index.html').exists() else 'in_progress'
    write(OUT/'status.json',status);return status

def acceptance_page(entries,complete=True):
    acceptance=OUT/'acceptance'
    def link(path):return quote(Path(os.path.relpath(path,acceptance)).as_posix(),safe='/')
    cards=[]
    for e in entries:
        event='跌倒' if e['dataset'] in ['gmd','fallvision'] else '打架'
        def label(value):return '无法判断' if value<0 else event if value==1 else '正常'
        scores=' / '.join(f'{mod}: '+label(d['prediction'])+('' if d['score'] is None else f'，分数 {d["score"]:.3f}') for mod,d in e['scores'].items())
        outcome=' '.join('unknown' if d['prediction']<0 else 'correct' if d['prediction']==e['label'] else 'false_alarm' if d['prediction']==1 else 'miss' for d in e['scores'].values())
        search=f'{e["dataset"]} {e["split"]} {outcome} {"normal" if e["label"]==0 else "event"}'
        source=link(e['source'])
        cards.append(f'<article data-tags="{search}"><h3>{e["dataset"]} · {e["split"]} · 标签：{label(e["label"])}</h3><p>{html.escape(scores)}</p><p><small>{e["sample_id"]}；左侧原画面，右侧观测骨架</small></p><video controls preload="none" src="{link(e["video"])}"></video><details><summary>动作序列图</summary><img loading="lazy" src="{link(e["poster"])}"></details><details><summary>查看完整原视频</summary><p>上方分数针对清单中的整段视频；骨架演示显示其中一个采样窗口。请结合原视频验收动作标签。</p><video controls preload="none" src="{source}"></video><a href="{source}">打开原视频文件</a><p>{html.escape(e["source"])}</p></details></article>')
    status='训练及外部评测已完成，等待人工验收。' if complete else '内部验收视频已完成；外部提取和评测仍在运行。'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>骨架与 RGB 验收</title><style>body{max-width:1400px;margin:24px auto;padding:0 16px;font:16px system-ui;background:#f4f6fa;color:#182536}h1{font-size:28px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:16px}article{background:white;padding:16px;border-radius:12px}video,img{width:100%}p{overflow-wrap:anywhere;color:#45546a}select{font:inherit;margin:8px 8px 16px 0;padding:8px}details{margin-top:12px}article[hidden]{display:none}</style><h1>骨架与 RGB 对照：验收视频</h1>'''
    page+=f'<p>{status}</p><p>显示固定种子 42 的视频级预测，分数不是经过校准的发生概率。骨架表示观察到的人体，不指认每个人的行为。TNUE 标签为 AI 暂定。完整三种子指标见 README。</p><label>数据集 <select id="dataset"><option value="">全部</option>'+''.join(f'<option>{n}</option>' for n in ['gmd','tnue','fallvision','vfd'])+'</select></label><label>样例 <select id="outcome"><option value="">全部</option><option value="normal">正常对照</option><option value="event">目标事件</option><option value="false_alarm">包含误报</option><option value="miss">包含漏报</option><option value="unknown">包含无法判断</option></select></label><span id="count"></span><div class="grid">'+''.join(cards)+'</div>'
    page+='''<script>const ds=document.querySelector('#dataset'),out=document.querySelector('#outcome');function update(){let count=0;document.querySelectorAll('article').forEach(e=>{const tags=e.dataset.tags.split(' ');e.hidden=!!((ds.value&&!tags.includes(ds.value))||(out.value&&!tags.includes(out.value)));if(!e.hidden)count++});document.querySelector('#count').textContent=count+' 段'}ds.onchange=out.onchange=update;update()</script></html>'''
    target=acceptance/('index.html' if complete else 'internal.html')
    target.write_text(page,encoding='utf-8')

def build_condition_analysis(summaries,folder):
    lines=['# 条件分层与失败分析','',
        '所有分数均来自固定的三个训练种子；外部标签和结果未用于重新选阈值或模型。宏平均 F1 的差异是本次配置的描述性比较，不代表对所有背景、人群或机位的保证。','']
    translations={'origin':'动作来源','person_size':'人物相对画面高度','pose_visibility_proxy':'关键点可见度代理','crowd':'同时观察到的人数'}
    for task in ['gmd','tnue']:
        summary=summaries[task];name='跌倒 / FallVision' if task=='gmd' else '打架 / VFD-2000'
        rgb=summary['aggregate']['rgb']['macro_f1'];sk=summary['aggregate']['skeleton']['macro_f1']
        cm=np.asarray(summary['per_seed']['skeleton_42']['confusion_matrix'])
        lines += [f'## {name}','',f'RGB 宏平均 F1 为 {rgb["mean"]:.3f} ± {rgb["std"]:.3f}，骨架为 {sk["mean"]:.3f} ± {sk["std"]:.3f}。骨架减 RGB 的均值差为 {sk["mean"]-rgb["mean"]:+.3f}。',
            f'骨架无法判断 {int(cm[:,2].sum())}/{summary["samples"]} 段，其中正常 {int(cm[0,2])} 段、目标事件 {int(cm[1,2])} 段；已全部保留在总数中。','',
            '| 分层 | 条件 | 正常 / 事件样本 | RGB 宏 F1 | 骨架宏 F1 | 骨架覆盖率 |','|---|---|---:|---:|---:|---:|']
        for attr,title in translations.items():
            for value in summary['strata']['rgb_42'][attr]:
                count=np.asarray(summary['strata']['rgb_42'][attr][value]['confusion_matrix']).sum(1)
                values={mod:[summary['strata'][f'{mod}_{seed}'][attr][value] for seed in [42,43,44]] for mod in ['rgb','skeleton']}
                f1={mod:f'{np.mean([m["macro_f1"] for m in ms]):.3f}' if count.min()>0 else '—（仅一类）' for mod,ms in values.items()}
                coverage=np.mean([m['coverage'] for m in values['skeleton']])
                lines.append(f'| {title} | {value} | {count[0]} / {count[1]} | {f1["rgb"]} | {f1["skeleton"]} | {coverage:.1%} |')
        reasons={}
        for path in (OUT/'evaluations'/task/'samples').glob('*.json'):
            result=read(path)['results']['skeleton_42']
            if result['prediction']<0:
                for reason in result['unknown_reasons']:reasons[reason]=reasons.get(reason,0)+1
        lines+=['','无法判断原因（同一片段可能有多个窗口原因）：'+('；'.join(f'{key}: {value}' for key,value in sorted(reasons.items())) or '无')+'。','']
    lines+=['## 解释边界','',
        '- `person_size`：相对高度小于 0.15 为 small，小于 0.35 为 medium，其余为 large；没有有效人体时高度取 0，small 组也包含提取失败案例。',
        '- 可见度以“至少一人满足关键点要求”的采样帧比例近似：低于 0.5 为 low，低于 0.9 为 partial，其余为 high。它不能直接证明真实遮挡，也不能保证第二个人清楚可见。',
        '- `multiple` 表示同一采样时刻检测到至少 3 人；它衡量多人场景，未人工标注人与人的具体遮挡关系。',
        '- `origin=bed` 的 FallVision 正样本已隔离，该组只用于正常片段误报核查；不据此报告床上跌倒召回率。',
        '- 轨迹不足会直接降低骨架覆盖率；低覆盖率时，应同时看无法判断数量，不能仅据较低的正常误报率判定更好。',
        '- 内部 GMD 测试仅 31 段，TNUE 仅 11 段；TNUE 标签仍为 AI 暂定。内部与外部样本组成不同，差异不能仅归因于服装、肤色或背景。',
        '- 原视频及每个窗口的分数、观测缺失位置保存在逐片段 JSON 中。实际家庭与教室机位测试待验收后进行。']
    (folder/'condition_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def build():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    acceptance=OUT/'acceptance';acceptance.mkdir(parents=True,exist_ok=True)
    models={};test_indices={};summaries={}
    for name in ['gmd','tnue']:
        for modality in ['rgb','skeleton']:
            for seed in [42,43,44]:
                base=OUT/name/modality/f'seed_{seed}';models[(name,modality,seed)]=read(base/'selection.json')
                test_indices[(name,modality,seed)]={r['sample_id']:r for r in read(base/'test_predictions.json')['predictions']}
        summaries[name]=read(OUT/'evaluations'/name/'summary.json')
        if summaries[name]['status']!='complete':raise RuntimeError('External evaluation incomplete')
    build_condition_analysis(summaries,acceptance)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ri,name in enumerate(['gmd','tnue']):
        for ci,modality in enumerate(['rgb','skeleton']):
            ax=axes[ri,ci]
            for seed in [42,43,44]:
                record=read(OUT/name/modality/f'seed_{seed}/run_record.json')
                for stage,values in record['stages'].items():
                    h=values['history'];ax.plot([r['epoch'] for r in h],[r['validation']['macro_f1'] for r in h],label=f'{seed} {stage}',linestyle='--' if stage=='finetune' else '-')
            ax.set(title=f'{name.upper()} / {modality}',xlabel='Epoch',ylabel='Validation macro F1',ylim=(0,1.02));ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(acceptance/'validation_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ri,name in enumerate(['gmd','tnue']):
        for ci,modality in enumerate(['rgb','skeleton']):
            cm=np.array(summaries[name]['per_seed'][f'{modality}_42']['confusion_matrix']);ax=axes[ri,ci];ax.imshow(cm,cmap='Blues')
            for i in range(2):
                for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
            ax.set(title=f'{name} external / {modality} / seed 42',xticks=[0,1,2],xticklabels=['normal','event','unknown'],yticks=[0,1],yticklabels=['normal','event'],xlabel='Prediction',ylabel='Source label')
    fig.savefig(acceptance/'external_confusion.png',dpi=160);plt.close(fig)
    lines=['# 骨架与 RGB 对照实验验收','',
           '训练与外部评测已完成；结果是否可接受由人工验收决定。视频中的骨架表示可见人体观测，事件分数针对视频片段，并不指认每个人的行为。','',
           '## 三个随机种子的结果','',
           '| 任务 | 模态 | 内部宏平均 F1（均值±标准差） | 外部宏平均 F1（均值±标准差） | 外部事件精确率 | 外部事件召回率 | 外部覆盖率 | 正常片段误报比例 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for name in ['gmd','tnue']:
        for mod in ['rgb','skeleton']:
            internal=[models[(name,mod,s)]['test']['macro_f1'] for s in [42,43,44]];ext=summaries[name]['aggregate'][mod]
            lines.append(f'| {"跌倒" if name=="gmd" else "打架"} | {mod} | {np.mean(internal):.3f} ± {np.std(internal,ddof=1):.3f} | {ext["macro_f1"]["mean"]:.3f} ± {ext["macro_f1"]["std"]:.3f} | {ext["precision"]["mean"][1]:.1%} | {ext["recall"]["mean"][1]:.1%} | {ext["coverage"]["mean"]:.1%} | {ext["normal_false_positive_rate"]["mean"]:.1%} |')
    lines+=['','## 范围与限制','',
            '- GMD 保留 145 段、排除 16 段；TNUE 保留 100 段，标签为 AI 暂定，仍需人工确认。',
            '- FallVision 使用 4,686 段筛选子集；956 段床相关正样本因落点范围未逐条确认而隔离，子集结果不可直接与此前全量结果比较。',
            '- FallVision 椅子／站立类别做了分层落点抽查，未逐条进行人工确认；原始类别标签仍可能含噪声。VFD 使用此前去重并排除已发现来源重叠后的 2,343 段。',
            '- 骨架提取不足记为 unknown，并进入总样本分母。遮挡分层采用关键点可见度代理量，不等同于人工遮挡标注。',
            '- 正常对照包含弯腰、坐下、主动躺下、抬臂运动及俯卧撑；GMD 的 81 段负样本有源描述与八帧序列 AI 复核记录。击掌和拥抱尚无专门确认的对照标注，不能宣称全部正常动作均已覆盖。',
            '- 两条路线使用相同清单、窗口与采样时刻；RGB 使用本地预训练动作主干，骨架动作网络从随机权重训练，比较的是整套方案表现。',
            '- 家庭实时测试、连续每小时误报率、提醒延迟及教室全景效果尚未验收。报告和 PPT 待验收后修改。','',
            '## 逐片段结果与视频','',
            f'- [验收页面]({(acceptance/"index.html").as_posix()})',
            f'- [全部内部／外部错误及无法判断索引]({(acceptance/"errors.json").as_posix()})',
            f'- [来源、人物大小、可见度与多人条件分析]({(acceptance/"condition_analysis.md").as_posix()})',
            f'- [验证曲线]({(acceptance/"validation_curves.png").as_posix()})',
            f'- [外部混淆矩阵]({(acceptance/"external_confusion.png").as_posix()})']
    lines+=['','## 文件与复现','',
        f'- [新模型与日志目录]({OUT.as_posix()})：`gmd/` 与 `tnue/` 下按 `rgb/`、`skeleton/`、`seed_42..44/` 保存；最终权重为 `selected_best.pt`。',
        f'- [共用数据清单与骨架缓存]({CACHE.as_posix()})；每个样本保留原视频路径，不复制下载数据。',
        f'- [跌倒外部完整指标及分层结果]({(OUT/"evaluations/gmd/summary.json").as_posix()})',
        f'- [打架外部完整指标及分层结果]({(OUT/"evaluations/tnue/summary.json").as_posix()})',
        f'- [运行环境和版本]({(OUT/"environment.json").as_posix()})；[复现命令]({(ROOT/"scripts/SKELETON_EXPERIMENT.md").as_posix()})。',
        f'- [权重、划分与预测一致性核对]({(OUT/"final_verification.json").as_posix()})；[断点恢复核对]({(OUT/"resume_verification.json").as_posix()})。']
    entries=[];errors=[]
    for name in ['gmd','tnue']:
        manifest=read(CACHE/'manifests'/f'{name}.json')
        for mod in ['rgb','skeleton']:
            for seed in [42,43,44]:
                for value in test_indices[(name,mod,seed)].values():
                    if value['prediction']!=value['label']:
                        errors.append(dict(dataset=name,split='internal_test',model=f'{mod}_{seed}',
                            **value,record=str(OUT/name/mod/f'seed_{seed}/test_predictions.json')))
        for row in manifest['rows']:
            if row['split']!='test':continue
            scores={mod:test_indices[(name,mod,42)][row['sample_id']] for mod in ['rgb','skeleton']}
            cache=torch.load(CACHE/name/(row['sample_id']+'.pt'),weights_only=True)
            media=render(row,cache,acceptance/'videos'/name,scores)
            entries.append(dict(dataset=name,split='internal_test',label=row['label'],sample_id=row['sample_id'],scores=scores,**media))
        ext='fallvision' if name=='gmd' else 'vfd';ext_rows={r['sample_id']:r for r in read(CACHE/'manifests'/f'{ext}.json')['rows']}
        buckets={}
        for path in sorted((OUT/'evaluations'/name/'samples').glob('*.json')):
            record=read(path);sid=record['sample_id']
            for key,value in record['results'].items():
                if value['prediction']!=record['label']:errors.append(dict(dataset=ext,split='external_test',sample_id=sid,path=record['path'],label=record['label'],model=key,prediction=value['prediction'],score=value['score'],unknown_reasons=value['unknown_reasons'],record=str(path)))
            for mod in ['rgb','skeleton']:
                pred=record['results'][f'{mod}_42']['prediction'];bucket=(mod,record['label'],pred)
                if len(buckets.get(bucket,[]))<2:buckets.setdefault(bucket,[]).append(record)
        chosen={r['sample_id']:r for records in buckets.values() for r in records}
        for sid,record in chosen.items():
            row=ext_rows[sid];cache_path=CACHE/ext/(sid+'.pt')
            if not cache_path.exists():continue
            cache=torch.load(cache_path,weights_only=True);scores={mod:record['results'][f'{mod}_42'] for mod in ['rgb','skeleton']}
            window=scores['skeleton']['peak_window']
            if window is None:window=scores['rgb']['peak_window'] or 0
            media=render(row,cache,acceptance/'videos'/ext,scores,window)
            entries.append(dict(dataset=ext,split='external_test',label=row['label'],sample_id=sid,scores=scores,**media))
    write(acceptance/'errors.json',errors);write(acceptance/'videos.json',entries)
    acceptance_page(entries);(OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    make_status();print('acceptance ready',len(entries),'videos;',len(errors),'error records',flush=True)

if __name__=='__main__':
    offline()
    p=argparse.ArgumentParser();p.add_argument('step',choices=['status','preview','internal','build']);a=p.parse_args()
    if a.step=='status':print(make_status())
    elif a.step=='preview':
        row=next(r for r in read(CACHE/'manifests/gmd.json')['rows'] if r['split']=='test' and r['label']==1)
        c=torch.load(CACHE/'gmd'/(row['sample_id']+'.pt'),weights_only=True);print(render(row,c,OUT/'preview'))
    elif a.step=='internal':
        all_media=[]
        for name in ['gmd','tnue']:
            predictions={mod:{r['sample_id']:r for r in read(OUT/name/mod/'seed_42/test_predictions.json')['predictions']} for mod in ['rgb','skeleton']}
            for row in read(CACHE/'manifests'/f'{name}.json')['rows']:
                if row['split']!='test':continue
                c=torch.load(CACHE/name/(row['sample_id']+'.pt'),weights_only=True)
                scores={mod:predictions[mod][row['sample_id']] for mod in predictions}
                all_media.append(dict(dataset=name,split='internal_test',label=row['label'],sample_id=row['sample_id'],scores=scores,**render(row,c,OUT/'acceptance/videos'/name,scores)))
            print('internal videos ready',name,flush=True)
        write(OUT/'acceptance/internal_videos.json',all_media)
        acceptance_page(all_media,complete=False)
    else:build()
