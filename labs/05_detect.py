# 실습 5: 사진에서 객체 검출(YOLO11n 또는 YOLO26n, COCO 80종)
#   python3 labs/05_detect.py <이미지> [yolo11n|yolo26n] [-o 결과.jpg]
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import COCO, draw_boxes, letterbox, model, to_npu, yolo11_decode, yolo26_decode
from dx_engine import InferenceEngine

args = [a for a in sys.argv[1:] if a != "-o"]
out = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else "detect.jpg"
if out in args:
    args.remove(out)
path, kind = args[0], (args[1] if len(args) > 1 else "yolo11n")
ie = InferenceEngine(model({"yolo11n": "yolo11-n_640x640.dxnn", "yolo26n": "yolo26-n_640x640.dxnn"}[kind]))

img = cv2.imread(path)
x, r, off = letterbox(img, 640)
pred = ie.run([to_npu(x)])[0]
dets = (yolo11_decode if kind == "yolo11n" else yolo26_decode)(pred, r, off, conf=0.3)

counts = {}
for *_, s, c in dets:
    counts[COCO[int(c)]] = counts.get(COCO[int(c)], 0) + 1
print(f"{len(dets)}개 검출:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
cv2.imwrite(out, draw_boxes(img, dets))
print("저장:", out)
