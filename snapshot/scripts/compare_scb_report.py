from pathlib import Path
import json,hashlib,time
from ultralytics import YOLO

def main():
 out=Path('results/detection/scb_comparison_20260913');out.mkdir(parents=True,exist_ok=True)
 data=Path('experiments/runs/scb/data/data.yaml').resolve()
 record={'purpose':'同一SCB测试清单上比较A和B；行为标签并非完整人员真值','data':str(data),'test_sha256':hashlib.sha256(data.with_name('test.txt').read_bytes()).hexdigest(),'parameters':{'imgsz':640,'device':0,'classes':[0],'conf':.001,'iou':.7,'max_det':300},'models':{}}
 for name,weights in [('A','models/yolov8n.pt'),('B','experiments/runs/scb/train/weights/best.pt')]:
  m=YOLO(weights).val(data=str(data),split='test',imgsz=640,device=0,classes=[0],conf=.001,iou=.7,max_det=300,batch=16,workers=0,plots=True,project=str(out.resolve()),name=name,verbose=False)
  ids=list(m.box.ap_class_index);idx=ids.index(0)
  p=float(m.box.p[idx]);r=float(m.box.r[idx]);ap=float(m.box.ap50[idx]);ap95=float(m.box.ap[idx])
  record['models'][name]={'weights':weights,'weights_sha256':hashlib.sha256(Path(weights).read_bytes()).hexdigest(),'precision':p,'recall':r,'f1':2*p*r/(p+r),'map50':ap,'map50_95':ap95,'speed_ms':m.speed}
  (out/'metrics.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),'utf-8')
 print(json.dumps(record,ensure_ascii=False))
if __name__=='__main__':main()
