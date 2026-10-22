# 실습 5: 사진에서 객체 검출(YOLO11n 또는 YOLO26n, COCO 80종)
#   python3 labs/05_detect.py <이미지> [--model yolo26n] [--conf 0.3] [-o 결과.jpg]
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import COCO, draw_boxes, letterbox, model, to_npu, yolo11_decode, yolo26_decode
from dx_engine import InferenceEngine

MODELS = {"yolo11n": "yolo11-n_640x640.dxnn", "yolo26n": "yolo26-n_640x640.dxnn"}
ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--model", choices=list(MODELS), default="yolo11n")
ap.add_argument("--conf", type=float, default=0.3, help="이보다 점수가 낮은 상자는 버린다")
ap.add_argument("-o", default="detect.jpg", help="결과 그림")
a = ap.parse_args()
path, out = a.image, a.o
ie = InferenceEngine(model(MODELS[a.model]))

img = cv2.imread(path)
x, r, off = letterbox(img, 640)
pred = ie.run([to_npu(x)])[0]
decode = yolo11_decode if a.model == "yolo11n" else yolo26_decode
dets = decode(pred, r, off, conf=a.conf)

counts = {}
for *_, c in dets:
    counts[COCO[int(c)]] = counts.get(COCO[int(c)], 0) + 1
print(
    f"{len(dets)}개 검출:",
    ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])),
)
cv2.imwrite(out, draw_boxes(img, dets))
print("저장:", out)
