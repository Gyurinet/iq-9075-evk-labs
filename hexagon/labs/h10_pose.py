# 실습 10: 두 단계 모델로 자세 추정.
# 사람을 찾고(FootTrackNet) 사람마다 관절 17개를 찾는다(HRNet)
#   python3 hexagon/labs/h10_pose.py samples/ppe/road_crew.jpg [-o pose.jpg]
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qnpu import Model, blur, people, resize_pad, to_rgb

# COCO 관절 연결(코·눈·귀, 어깨·팔, 엉덩이·다리)
# fmt: off
LINKS = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
         (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
# fmt: on

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("-o", default="pose.jpg")
a = ap.parse_args()

finder = Model("people")
pose = Model("pose")  # 입력 256×192(세로로 긴 사람 한 명)
img = cv2.imread(a.image)
H, W = img.shape[:2]

t0 = time.perf_counter()
x, r, off = resize_pad(img, *finder.size)
faces, persons, _ = people(finder.run(to_rgb(x)), r, off)  # ① 사람 찾기
t1 = time.perf_counter()
skeletons = []
for x1, y1, x2, y2, _ in persons:
    # ② 사람 상자를 3:4 비율로 넓혀 자르고 256×192로 줄인다
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h = max(y2 - y1, (x2 - x1) * 4 / 3) * 1.2
    w = h * 3 / 4
    src = np.float32(
        [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx - w / 2, cy + h / 2]]
    )
    dst = np.float32([[0, 0], [192, 0], [0, 256]])
    crop = cv2.warpAffine(img, cv2.getAffineTransform(src, dst), (192, 256))
    hm = pose.run(to_rgb(crop))["heatmaps"]  # [64, 48, 17]
    # ③ 관절마다 열지도의 가장 높은 칸 → 원래 사진 좌표(열지도 1칸 = 4화소)
    flat = hm.reshape(-1, 17)
    idx = flat.argmax(0)
    py, px = np.unravel_index(idx, hm.shape[:2])
    score = flat.max(0)
    kx = cx - w / 2 + (px + 0.5) * 4 * w / 192
    ky = cy - h / 2 + (py + 0.5) * 4 * h / 256
    skeletons.append(np.stack([kx, ky, score], 1))
t2 = time.perf_counter()
print(
    f"사람 {len(persons)}명: 사람 찾기 {(t1 - t0) * 1000:.1f} ms,"
    f" 관절 {(t2 - t1) * 1000:.1f} ms"
)

blur(img, faces)
t = max(2, round(min(H, W) / 300))
for k in skeletons:
    ok = k[:, 2] >= 0.3
    for i, j in LINKS:
        if ok[i] and ok[j]:
            p, q = tuple(k[i, :2].astype(int)), tuple(k[j, :2].astype(int))
            cv2.line(img, p, q, (0, 200, 255), t, cv2.LINE_AA)
    for px, py, _ in k[ok]:
        cv2.circle(img, (int(px), int(py)), t + 1, (255, 255, 255), -1)
cv2.imwrite(a.o, img)
print("저장:", a.o)
