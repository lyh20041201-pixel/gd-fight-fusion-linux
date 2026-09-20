"""Authenticated local pipe worker; retains the original inference environment."""
from pathlib import Path
import sys, os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false')
import argparse
from multiprocessing.connection import Listener
from backend.vision.live_actions import LiveActionModels

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pipe',required=True)
    parser.add_argument('--manifest',required=True)
    args=parser.parse_args()
    models=LiveActionModels(args.manifest)
    with Listener(args.pipe,family='AF_PIPE',authkey=bytes.fromhex(os.environ.pop('GD_ACTION_AUTH'))) as listener:
        with listener.accept() as conn:
            import torch
            from backend.vision.action_guards import GUARD_VERSION
            conn.send(dict(ready=True,torch=torch.__version__,model_version=models.version,
                fall_model=models.manifest['fall']['label'],fall_architecture=models.fall_kind,
                fight_model=models.manifest['fight']['label'],guard_version=GUARD_VERSION))
            while True:
                try: request=conn.recv()
                except EOFError: break
                if request.get('op')=='stop': break
                try:
                    if request.get('op')=='pose_preview':
                        conn.send(models.pose_preview(request['frames'][0]))
                    else:
                        conn.send(models.predict(request['camera'],request['frames']))
                except Exception as exc: conn.send(dict(state='error',reason=str(exc)[:250],actions=[]))

if __name__=='__main__':main()
