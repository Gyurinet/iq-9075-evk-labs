# 사람 오려내기: 화소마다 사람인지 알아내(DeepLabV3+, NPU) 배경이나 사람을 바꾼다
#   python3 hexagon/apps/cutout.py samples/ppe/loading.jpg -o cutout.jpg --mode blur-bg
#   python3 hexagon/apps/cutout.py samples/videos/pedestrians.mp4 --save hide.mp4 --mode hide
#   --mode blur-bg : 배경 흐리기(화상 회의, 제품·인물 사진)
#          remove-bg : 배경을 한 색으로(누끼, 자료 사진)
#          hide : 사람 전체를 실루엣으로 가리기(얼굴만 가리는 것보다 강한 개인정보 보호)
#   --cls person : 대상(PASCAL VOC 20종 중 하나. car, bus, bicycle, dog, chair, bottle 등)
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 저장소 맨 위(labkit)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hexagon/(qnpu)
from labkit import VideoSource, VideoWriter
from qnpu import Model, Pool, to_rgb

# fmt: off
VOC = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
       "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
       "sheep", "sofa", "train", "tvmonitor"]
# fmt: on
ap = argparse.ArgumentParser()
ap.add_argument("src", help="사진 또는 영상")
ap.add_argument("-o", default="cutout.jpg", help="결과 사진")
ap.add_argument("--save", default="cutout.mp4", help="결과 영상")
ap.add_argument("--mode", choices=["blur-bg", "remove-bg", "hide"], default="blur-bg")
ap.add_argument("--cls", default="person", choices=VOC[1:])
a = ap.parse_args()
TARGET = VOC.index(a.cls)


def mask_of(m, img):
    """사진 → 대상일 확률에 가까운 0~1 마스크(경계를 부드럽게)"""
    h, w = m.size  # 520×520
    cls = m.run(to_rgb(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)))["mask"]
    mask = (cls == TARGET).astype(np.float32)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(mask, (0, 0), 2)[..., None]


def apply(img, mask):
    if a.mode == "blur-bg":
        back = cv2.GaussianBlur(img, (0, 0), 15)
    elif a.mode == "remove-bg":
        back = np.full_like(img, (90, 160, 60))  # 초록 배경(BGR)
    else:  # hide: 사람 자리를 회색 실루엣으로
        return (img * (1 - mask) + np.full_like(img, 128) * mask).astype(np.uint8)
    return (img * mask + back * (1 - mask)).astype(np.uint8)


if a.src.lower().endswith((".jpg", ".jpeg", ".png")):
    m = Model("deeplab")
    img = cv2.imread(a.src)
    t0 = time.perf_counter()
    mask = mask_of(m, img)
    out = apply(img, mask)
    t1 = time.perf_counter()
    cv2.imwrite(a.o, out)
    print(
        f"{a.cls} 화소 {float((mask > 0.5).mean()) * 100:.0f}%, NPU {m.last_ms:.1f} ms,"
        f" 전체 {(t1 - t0) * 1000:.0f} ms → {a.o}"
    )
    sys.exit()

W, H = 1280, 720
pool = Pool("deeplab", 2)
video = VideoSource(a.src, (W, H))
writer = VideoWriter(a.save, (W, H), video.fps)
jobs = deque()
n = 0
t0 = time.perf_counter()
for frame in video:
    jobs.append((frame, pool.submit(mask_of, frame)))
    n += 1
    if len(jobs) >= 4:
        f, fut = jobs.popleft()
        writer.write(apply(f, fut.result()))
while jobs:
    f, fut = jobs.popleft()
    writer.write(apply(f, fut.result()))
wall = time.perf_counter() - t0
video.close()
pool.close()
writer.close()
print(f"{n}프레임을 {wall:.1f}초에 처리: {n / wall:.1f} FPS → {a.save}")
