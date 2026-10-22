# 실습 12: 화질 높이기(초해상도).
# 작은 사진을 4배로 키울 때 보통 확대와 NPU 모델(XLSR)을 비교한다
#   python3 hexagon/labs/h12_super_res.py samples/images/crossing.jpg [-o super_res.jpg]
#   원본을 4분의 1로 줄인 뒤 다시 4배로 키워, 원본과 얼마나 같은지(PSNR) 잰다
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qnpu import Model, to_rgb

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("-o", default="super_res.jpg")
a = ap.parse_args()

m = Model("sr")  # 128×128 → 512×512
T = 128  # 조각 크기


def upscale(small):
    """작은 사진을 128×128 조각으로 나눠 NPU로 4배씩 키우고 이어 붙인다"""
    h, w = small.shape[:2]
    ph, pw = -h % T, -w % T  # 128의 배수가 되게 가장자리를 늘린다
    pad = cv2.copyMakeBorder(small, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    out = np.zeros((pad.shape[0] * 4, pad.shape[1] * 4, 3), np.uint8)
    for y in range(0, pad.shape[0], T):
        for x in range(0, pad.shape[1], T):
            tile = m.run(to_rgb(pad[y : y + T, x : x + T]))["upscaled_image"]
            out[y * 4 : (y + T) * 4, x * 4 : (x + T) * 4] = np.clip(tile * 255, 0, 255)
    return cv2.cvtColor(out[: h * 4, : w * 4], cv2.COLOR_RGB2BGR)


def psnr(x, y):
    return cv2.PSNR(x, y)


img = cv2.imread(a.image)
h, w = (img.shape[0] // 4) * 4, (img.shape[1] // 4) * 4
img = img[:h, :w]
small = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)  # 저해상도 영상 흉내
t0 = time.perf_counter()
sr = upscale(small)
t1 = time.perf_counter()
bicubic = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
tiles = -(-small.shape[0] // T) * -(-small.shape[1] // T)
print(f"{small.shape[1]}×{small.shape[0]} → {w}×{h}: 조각 {tiles}개, {(t1 - t0) * 1000:.0f} ms")
print(
    f"원본과의 PSNR(높을수록 같음): 보통 확대 {psnr(bicubic, img):.2f} dB,"
    f" NPU {psnr(sr, img):.2f} dB"
)

# 결과: 가운데 일부를 잘라 3배로 키워 [보통 확대 | NPU | 원본]으로 나란히(차이가 잘 보이게)
cy, cx, s = h // 2, w // 2, min(h, w) // 10
crop = [im[cy - s : cy + s, cx - s : cx + s] for im in (bicubic, sr, img)]
crop = [cv2.resize(c, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST) for c in crop]
cv2.imwrite(a.o, np.hstack(crop))
print("저장:", a.o, "(왼쪽부터 보통 확대, NPU, 원본)")
