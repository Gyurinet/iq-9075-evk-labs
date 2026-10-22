# 실습 7: 사람 자세 추정(YOLO11n-pose). 사람마다 관절 17개를 찾는다
#   python3 labs/07_pose.py <이미지> [-o 결과.jpg]
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import letterbox, model, to_npu
from dx_engine import InferenceEngine

# COCO 관절 연결(코·눈·귀, 어깨·팔, 엉덩이·다리)
# fmt: off
LINKS = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
         (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
# fmt: on

path = sys.argv[1]
out = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else "pose.jpg"
ie = InferenceEngine(model("yolo11-n-pose_640x640.dxnn"))

img = cv2.imread(path)
x, r, (left, top) = letterbox(img, 640)
p = ie.run([to_npu(x)])[0][0].T  # [8400, 4 + 1 + 17×3]
keep = p[:, 4] >= 0.4
p = p[keep]
xywh = p[:, :4].copy()
xywh[:, :2] -= xywh[:, 2:] / 2
idx = cv2.dnn.NMSBoxes(xywh.tolist(), p[:, 4].tolist(), 0.4, 0.45)
idx = np.array(idx, int).reshape(-1)

t = max(2, round(min(img.shape[:2]) / 300))
for i in idx:
    k = p[i, 5:].reshape(17, 3)  # (x, y, 신뢰도)
    pts = (k[:, :2] - [left, top]) / r
    ok = k[:, 2] >= 0.5
    for a, b in LINKS:
        if ok[a] and ok[b]:
            cv2.line(
                img,
                tuple(pts[a].astype(int)),
                tuple(pts[b].astype(int)),
                (255, 160, 40),
                t,
                cv2.LINE_AA,
            )
    for j in np.where(ok)[0]:
        cv2.circle(img, tuple(pts[j].astype(int)), t + 1, (40, 220, 255), -1, cv2.LINE_AA)
print(f"사람 {len(idx)}명")
cv2.imwrite(out, img)
print("저장:", out)
