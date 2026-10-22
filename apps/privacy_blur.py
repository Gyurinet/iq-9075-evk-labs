# 영상 개인정보 보호: 얼굴을 찾아(NPU) 흐리게 가린다(ARM). 외부에 영상을 넘기기 전 처리에 쓴다
#   python3 apps/privacy_blur.py samples/videos/pedestrians.mp4 --save blurred.mp4
#   python3 apps/privacy_blur.py samples/images/pedestrians.jpg -o blurred.jpg     (사진 한 장)
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import VideoSource, VideoWriter, letterbox, model, to_npu, unletter
from dx_engine import InferenceEngine

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--save", help="결과 영상(mp4)")
ap.add_argument("-o", help="결과 사진(jpg)")
ap.add_argument("--conf", type=float, default=0.3)
ap.add_argument("--show", action="store_true", help="가린 자리에 상자도 그린다(확인용)")
a = ap.parse_args()

ie = InferenceEngine(model("yolov5-s-face_640x640.dxnn"))


def faces(pred, r, off):
    """YOLOv5-face 출력 [25200, 16] (cx, cy, w, h, 물체 점수, 눈·코·입 5점, 얼굴 점수) → 얼굴 상자"""
    p = pred.reshape(-1, 16)
    score = p[:, 4] * p[:, 15]
    p, score = p[score >= a.conf], score[score >= a.conf]
    if not len(p):
        return np.zeros((0, 4))
    xywh = p[:, :4].copy()
    xywh[:, :2] -= xywh[:, 2:] / 2
    idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), score.tolist(), a.conf, 0.4), int).reshape(-1)
    b = xywh[idx]
    b[:, 2:] += b[:, :2]
    return unletter(b, r, off)


def blur(img, boxes):
    """얼굴 상자를 조금 넓혀(머리카락·귀까지) 모자이크 + 흐림"""
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        x1, y1 = int(max(0, x1 - bw * 0.15)), int(max(0, y1 - bh * 0.2))
        x2, y2 = int(min(w, x2 + bw * 0.15)), int(min(h, y2 + bh * 0.1))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = img[y1:y2, x1:x2]
        small = cv2.resize(roi, (max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)), interpolation=cv2.INTER_AREA)
        img[y1:y2, x1:x2] = cv2.GaussianBlur(cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST), (0, 0), 3)
        if a.show:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 2)
    return img


if a.src.lower().endswith((".jpg", ".jpeg", ".png")):
    img = cv2.imread(a.src)
    x, r, off = letterbox(img, 640)
    found = faces(ie.run([to_npu(x)])[0], r, off)
    cv2.imwrite(a.o or "blurred.jpg", blur(img, found))
    print(f"얼굴 {len(found)}개를 가렸습니다 → {a.o or 'blurred.jpg'}")
    sys.exit()

W, H = 1280, 720
video = VideoSource(a.src, (W, H))
out = VideoWriter(a.save or "blurred.mp4", (W, H), video.fps)
jobs = deque()
n, total = 0, 0
t0 = time.perf_counter()


def finish(job):
    global total
    jid, frame, r, off = job
    found = faces(ie.wait(jid)[0], r, off)
    total += len(found)
    out.write(blur(frame, found))


for frame in video:
    x, r, off = letterbox(frame, 640)
    jobs.append((ie.run_async([to_npu(x)]), frame.copy(), r, off))
    n += 1
    if len(jobs) >= 4:
        finish(jobs.popleft())
while jobs:
    finish(jobs.popleft())
wall = time.perf_counter() - t0
video.close()
out.close()
print(f"{n}프레임을 {wall:.1f}초에 처리: {n / wall:.1f} FPS, 프레임당 얼굴 {total / n:.1f}개 → {a.save or 'blurred.mp4'}")
