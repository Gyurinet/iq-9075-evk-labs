# 실습 4: 객체 검출(YOLOX, COCO 80종)을 내장 NPU에서. 겹친 상자 정리(NMS)는 ARM이 한다
#   python3 hexagon/labs/h04_detect.py samples/images/crossing.jpg [-o detect.jpg] [--conf 0.4]
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 저장소 맨 위(labkit)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hexagon/(qnpu)
from labkit import COCO, draw_boxes
from qnpu import Model, resize_pad, to_rgb, yolox

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("-o", default="detect.jpg")
ap.add_argument("--conf", type=float, default=0.4)
a = ap.parse_args()

m = Model("yolox")
img = cv2.imread(a.image)
t0 = time.perf_counter()
x, r, off = resize_pad(img, *m.size)  # ARM: 640×640에 맞추기(남는 곳은 검게)
out = m.run(to_rgb(x))  # NPU: 상자 8400개 후보와 점수·분류
t1 = time.perf_counter()
dets = yolox(out, r, off, a.conf)  # ARM: 점수 거르기 + NMS + 원래 좌표로
t2 = time.perf_counter()
print(f"검출 {len(dets)}개:", ", ".join(sorted({COCO[int(d[5])] for d in dets})))
print(
    f"전처리+NPU {(t1 - t0) * 1000:.1f} ms (NPU {m.last_ms:.1f} ms),"
    f" 후처리 {(t2 - t1) * 1000:.1f} ms"
)
cv2.imwrite(a.o, draw_boxes(img, dets))
print("저장:", a.o)
