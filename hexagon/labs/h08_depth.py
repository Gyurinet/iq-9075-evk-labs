# 실습 8: 한 장의 사진으로 거리(깊이) 추정(Depth Anything V2).
# 실수 모델을 NPU에서 fp16으로 실행한다
#   python3 hexagon/labs/h08_depth.py samples/images/forklift.jpg [-o depth.jpg]
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qnpu import Model, to_rgb

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("-o", default="depth.jpg")
a = ap.parse_args()

m = Model("depth")  # 처음 한 번은 NPU용 그래프를 만드느라 10초 정도 걸린다
img = cv2.imread(a.image)
h, w = m.size  # 518×518
x = to_rgb(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)).astype(np.float32) / 255
d = m.run(x)["depth_estimates"][..., 0]  # 가까울수록 큰 값(상대 깊이)

lo, hi = np.percentile(d, [2, 98])  # 튀는 값은 잘라 내고 0~1로
d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
d = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
vis = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
cv2.imwrite(a.o, np.hstack([img, vis]))
print(f"모델 올리기 {m.load_s:.1f}초, NPU {m.last_ms:.1f} ms, 저장: {a.o}")
