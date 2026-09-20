"""Opt-in local pose preview. Does not classify actions, record video or emit alarms."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,time
import cv2,torch
from scripts.skeleton_common import POSE,POSE_SIZE,OUT,offline,read,sha


def main():
    parser=argparse.ArgumentParser(description='Local camera pose preview; Q or Esc closes the camera.')
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--device',default='cpu',help='CPU default keeps the offline experiment GPU available.')
    parser.add_argument('--source',type=Path,help='Use an existing local video instead of the camera.')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--max-frames',type=int)
    args=parser.parse_args()
    offline();torch.set_num_threads(4);cv2.setNumThreads(2)
    if not POSE.is_file() or POSE.stat().st_size!=POSE_SIZE or sha(POSE)!=read(OUT/'pose_provenance.json')['sha256']:
        raise ValueError('Verified local pose weight is missing or changed; download is disabled.')
    if args.source is not None and not args.source.is_file():raise ValueError('Source must be an existing local video.')
    from ultralytics import YOLO
    model=YOLO(str(POSE),task='pose')
    cap=cv2.VideoCapture(str(args.source)) if args.source else cv2.VideoCapture(args.camera,cv2.CAP_DSHOW)
    if not cap.isOpened():raise RuntimeError('Camera/video could not open. Close other camera applications or choose --camera 1.')
    if not args.source:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    title='Local pose preview - no action classification - Q / Esc to close'
    count=0;started=time.monotonic()
    try:
        while True:
            ok,frame=cap.read()
            if not ok:
                if not args.source:raise RuntimeError('Camera stopped returning frames.')
                break
            result=model.predict(frame,imgsz=640,conf=.25,iou=.45,max_det=100,device=args.device,
                                 verbose=False,save=False)[0]
            preview=result.plot(boxes=False,labels=False,probs=False,kpt_radius=3)
            cv2.putText(preview,'POSE ONLY | no recording | no fall/fight alarms',(10,24),0,.48,(0,230,255),1)
            count+=1
            if not args.headless:
                cv2.imshow(title,preview)
                if cv2.waitKey(1)&0xFF in [27,ord('q')] or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
            if args.max_frames and count>=args.max_frames:break
    finally:
        cap.release()
        if not args.headless:cv2.destroyAllWindows()
    print({'frames':count,'elapsed_seconds':round(time.monotonic()-started,2),
           'source':'local_video' if args.source else f'camera_{args.camera}',
           'mode':'pose_preview_only','recording':False},flush=True)


if __name__=='__main__':main()
