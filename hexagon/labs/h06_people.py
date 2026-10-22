# 실습 6: 사람·얼굴·머리와 발 위치를 한 번에(FootTrackNet). 얼굴은 흐리게 가려 저장한다
#   python3 hexagon/labs/h06_people.py samples/images/pedestrians.jpg [-o people.jpg]
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qnpu import Model, blur, people, resize_pad, to_rgb

POINTS = {0: (0, 200, 255), 15: (255, 120, 0), 16: (255, 120, 0)}  # 코(머리), 왼·오른 발목

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("-o", default="people.jpg")
a = ap.parse_args()

m = Model("people")
img = cv2.imread(a.image)
x, r, off = resize_pad(img, *m.size)  # 480×640
faces, persons, kps = people(m.run(to_rgb(x)), r, off)
print(f"사람 {len(persons)}명, 얼굴 {len(faces)}개 (NPU {m.last_ms:.1f} ms)")

blur(img, faces)
t = max(2, round(min(img.shape[:2]) / 300))
for (x1, y1, x2, y2, _), k in zip(persons, kps, strict=True):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (80, 200, 60), t)
    for i, c in POINTS.items():  # 이 모델은 머리와 발 위치를 찾도록 학습되었다
        px, py, v = k[i]
        if v >= 0.5:
            cv2.circle(img, (int(px), int(py)), t * 3, c, -1)
cv2.imwrite(a.o, img)
print("저장:", a.o)
