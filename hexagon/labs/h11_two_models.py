# 실습 11: 서로 다른 모델 두 개를 함께 쓰기.
# 영상마다 객체 검출(YOLOX)과 도로 장면 분할(FFNet)을 한다
#   python3 hexagon/labs/h11_two_models.py samples/videos/roundabout.mp4 [--save out.mp4]
#   [1] 두 모델을 HTP 0에 올려 차례로  [2] HTP 0·1에 하나씩 올려 동시에
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 저장소 맨 위(labkit)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hexagon/(qnpu)
from labkit import VideoSource, VideoWriter, cpu_seconds, draw_boxes
from qnpu import Model, resize_pad, to_rgb, yolox

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--save", help="[2]의 결과를 영상으로 저장")
ap.add_argument("--frames", type=int, default=200, help="처리할 프레임 수")
a = ap.parse_args()
SIZE = (1280, 720)
# Cityscapes 색 일부: 도로·인도만 칠한다
ROAD = np.zeros((19, 3), np.uint8)
ROAD[0], ROAD[1] = (128, 64, 128), (232, 35, 244)


def detect(m, frame):
    x, r, off = resize_pad(frame, *m.size)
    return yolox(m.run(to_rgb(x)), r, off)


def segment(m, frame):
    h, w = m.size
    x = to_rgb(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA))
    return m.run(x)["mask"].argmax(-1).astype(np.uint8)  # [128, 256]


def draw(frame, dets, cls):
    road = ROAD[cv2.resize(cls, SIZE, interpolation=cv2.INTER_NEAREST)]
    mixed = np.where(
        road.any(-1, keepdims=True), cv2.addWeighted(frame, 0.5, road, 0.5, 0), frame
    )
    return draw_boxes(mixed, dets)


def run(det_m, seg_m, together, save=None):
    video = VideoSource(a.video, SIZE)
    out = VideoWriter(save, SIZE, video.fps) if save else None
    pool = ThreadPoolExecutor(2)
    n = 0
    c0, t0 = cpu_seconds(), time.perf_counter()
    for frame in video:
        if together:  # 두 모델을 동시에 맡기고 둘 다 끝나기를 기다린다
            f1, f2 = pool.submit(detect, det_m, frame), pool.submit(segment, seg_m, frame)
            dets, cls = f1.result(), f2.result()
        else:
            dets, cls = detect(det_m, frame), segment(seg_m, frame)
        if out:
            out.write(draw(frame, dets, cls))
        n += 1
        if n >= a.frames:
            break
    wall = time.perf_counter() - t0
    video.close()
    pool.shutdown()
    if out:
        out.close()
    return n / wall, (cpu_seconds() - c0) / wall


print("모델: YOLOX(검출) + FFNet-40S(분할), 최고 성능 모드")
fps, cores = run(
    Model("yolox", htp=0, perf="burst"), Model("ffnet", htp=0, perf="burst"), False
)
print(f"[1] HTP 0 하나에서 차례로 : {fps:6.1f} FPS, CPU {cores:.2f}코어")
det, seg = Model("yolox", htp=0, perf="burst"), Model("ffnet", htp=1, perf="burst")
fps, cores = run(det, seg, True)
print(f"[2] HTP 0·1에서 동시에   : {fps:6.1f} FPS, CPU {cores:.2f}코어")
if a.save:  # 결과 영상은 따로 한 번 더 돌려 저장한다(저장 시간이 측정에 섞이지 않게)
    run(det, seg, True, a.save)
    print("결과 영상:", a.save)
