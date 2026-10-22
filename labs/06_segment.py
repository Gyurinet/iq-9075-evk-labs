# 실습 6: 인스턴스 분할(YOLO11n-seg). 상자 안에서 물체 모양(마스크)까지 칠한다
#   python3 labs/06_segment.py <이미지> [-o 결과.jpg]
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import COCO, color, draw_boxes, letterbox, model, to_npu
from dx_engine import InferenceEngine

path = sys.argv[1]
out = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else "segment.jpg"
ie = InferenceEngine(model("yolo11-n-seg_640x640.dxnn"))

img = cv2.imread(path)
x, r, (left, top) = letterbox(img, 640)
pred, proto = ie.run([to_npu(x)])                         # [1,116,8400], [1,32,160,160]
p = pred[0].T                                             # [8400, 4+80+32]
cls = p[:, 4:84].argmax(1)
score = p[np.arange(len(p)), 4 + cls]
keep = score >= 0.3
p, cls, score = p[keep], cls[keep], score[keep]
xywh = p[:, :4].copy()
xywh[:, :2] -= xywh[:, 2:] / 2
idx = np.array(cv2.dnn.NMSBoxesBatched(xywh.tolist(), score.tolist(), cls.tolist(), 0.3, 0.45), int).reshape(-1)

# 마스크 = 계수(32) × 원형 마스크(32×160×160). ARM에서 행렬 곱 한 번
coef = p[idx, 84:]
masks = 1 / (1 + np.exp(-(coef @ proto[0].reshape(32, -1)))).reshape(-1, 160, 160)
h, w = img.shape[:2]
nh, nw = round(h * r), round(w * r)
overlay = img.copy()
boxes = []
for k, i in enumerate(idx):
    m = cv2.resize(masks[k], (640, 640))[top:top + nh, left:left + nw]
    m = cv2.resize(m, (w, h)) > 0.5
    x1, y1, bw, bh = xywh[i]
    box = np.array([x1 - left, y1 - top, x1 + bw - left, y1 + bh - top]) / r
    clip = np.zeros_like(m)
    clip[max(0, int(box[1])):int(box[3]), max(0, int(box[0])):int(box[2])] = True
    overlay[m & clip] = color(cls[i])
    boxes.append([*box, score[i], cls[i]])
img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
draw_boxes(img, np.array(boxes).reshape(-1, 6))
print(f"{len(idx)}개:", ", ".join(sorted({COCO[int(cls[i])] for i in idx})))
cv2.imwrite(out, img)
print("저장:", out)
