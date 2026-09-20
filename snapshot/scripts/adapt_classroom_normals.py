"""Second, predeclared mix: Le2i + reviewed real-classroom normal snippets.

This is not a blind model comparison: the first pilot was already inspected.
All snippets from the Cambridge lesson belong to training, never validation/test.
"""
import copy
import json
import subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import imageio_ffmpeg
import torch
from ultralytics import YOLO
import scripts.train_classroom_posec3d as pilot
from scripts.prepare_classroom_posec3d import sha,write


def main():
    pilot.setup()
    original_out=pilot.OUT
    pilot.OUT=original_out/'classroom_mix'
    pilot.CANDIDATE=ROOT/'models/classroom_posec3d/classroom_mix_head_pilot.pth'
    pilot.POLICY=copy.deepcopy(pilot.POLICY)
    pilot.POLICY.update(version=2,scope='Le2i primary actor + all eligible tracks in classroom normal snippets',
        additional_data='33.92s from one Cambridge classroom lesson, AI-reviewed normal; no independent dense human labels',
        source_grouping='All Cambridge clips train only; equal weight as ONE original source per binary class',
        comparison_caveat='Le2i test results were seen in the first pilot; this is a follow-up exploratory experiment, not a new blind test')
    rows=json.loads((original_out/'manifest.json').read_text(encoding='utf-8'))['videos']
    source=ROOT/'datasets/classroom_observation/cambridge_lesson.mp4'
    specs=[(101,545.,554.5),(102,555.,560.),(103,595.,603.8),(104,604.4,610.),(105,651.4,656.4)]
    for index,start,end in specs:
        out=ROOT/f'datasets/classroom_observation/clips/normal_{index}.mp4';out.parent.mkdir(parents=True,exist_ok=True)
        if not out.exists():
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-ss',str(start),'-i',str(source),'-t',str(end-start),
                '-an','-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True)
        cap=cv2.VideoCapture(str(out));fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
        shape=[int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))];cap.release()
        rows.append(dict(id=index,split='train',subject='Cambridge classroom group',source_group='cambridge_lesson',
            video=str(out.relative_to(ROOT)),video_sha256=sha(out),source_sha256=sha(source),fps=fps,frames=frames,shape=shape,
            duration=frames/fps,source_start=start,source_end=end,spans=[],reviewed_normal=True,multi_person=True,
            label_provenance='AI visual review of ~3.1fps contact sheets: teaching, walking, bending, seated work; excludes observed shot changes',
            label_limitation='Weak video-level normal annotation; not independent human ground truth; obscured bodies cannot be fully verified'))
    write(pilot.OUT/'policy.json',pilot.POLICY)
    write(pilot.OUT/'manifest.json',dict(videos=rows))
    detector=YOLO(str(ROOT/'models/yolov8n-pose.pt'),task='pose')
    from ultralytics.utils import LOGGER
    LOGGER.setLevel('ERROR')
    for row in rows:
        if row['id']>=100:
            pilot.extract_pose(row,detector)
    del detector;torch.cuda.empty_cache()
    model=pilot.load_posec3d(pilot.BASE)
    for row in rows:
        if row['id']>=100:
            pilot.feature_windows(row,model)
    pilot.train(rows,model)


if __name__=='__main__':
    main()
