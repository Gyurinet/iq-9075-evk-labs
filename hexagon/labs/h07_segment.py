# 실습 7: 도로 장면 분할(FFNet-40S, Cityscapes 19종). 화소마다 도로·인도·차·사람 등을 칠한다
#   python3 hexagon/labs/h07_segment.py samples/images/roundabout.jpg [-o segment.jpg]
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qnpu import Model, labels, to_rgb

# Cityscapes 공식 색(BGR로 바꿔 둠)
# fmt: off
RGB = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
       (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
       (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
       (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]
COLORS = np.array(RGB, np.uint8)[:, ::-1]
# fmt: on

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("-o", default="segment.jpg")
a = ap.parse_args()

names = labels("ffnet")
m = Model("ffnet")
img = cv2.imread(a.image)
h, w = m.size  # 1024×2048. 가로세로 비율이 달라도 그대로 늘려 넣는다(학습 방식과 같게)
out = m.run(to_rgb(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)))[
    "mask"
]  # [128, 256, 19]
cls = out.argmax(-1).astype(np.uint8)  # ARM: 화소마다 가장 높은 분류
cls = cv2.resize(cls, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
vis = cv2.addWeighted(img, 0.45, COLORS[cls], 0.55, 0)

share = np.bincount(cls.ravel(), minlength=19) / cls.size
top = np.argsort(share)[::-1][:6]
print(
    f"NPU {m.last_ms:.1f} ms | " + ", ".join(f"{names[i]} {share[i] * 100:.0f}%" for i in top)
)
cv2.imwrite(a.o, vis)
print("저장:", a.o)
