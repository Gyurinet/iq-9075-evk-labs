# 통행량 세기: 영상 속 사람·자전거·차량이 가상의 선을 어느 방향으로 지나가는지 센다
#   python3 apps/line_counter.py samples/videos/pedestrians.mp4 --save counted.mp4
#   --line x1,y1,x2,y2 : 선의 양 끝(화면 폭·높이에 대한 비율 0~1). 고정된 카메라 영상에서 쓴다
#   방향: 세로선이면 left/right(왼쪽으로·오른쪽으로), 가로선이면 up/down(멀어짐·다가옴)
#   결과: 화면 표시 영상(--save), 1분 단위 집계 CSV(--csv, 기본 counts.csv)
import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import (
    COCO,
    Tracker,
    VideoSource,
    VideoWriter,
    color,
    letterbox,
    model,
    put_text,
    side,
    to_npu,
    yolo26_decode,
)
from dx_engine import InferenceEngine

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--line", default="0.5,0.3,0.5,1")
ap.add_argument("--classes", default="person,bicycle,car,bus,truck")
ap.add_argument("--save")
ap.add_argument("--csv", default="counts.csv")
ap.add_argument("--conf", type=float, default=0.35)
a = ap.parse_args()

W, H = 1280, 720
x1, y1, x2, y2 = (float(v) for v in a.line.split(","))
A, B = (x1 * W, y1 * H), (x2 * W, y2 * H)
want = [COCO.index(c) for c in a.classes.split(",")]
vertical = abs(B[0] - A[0]) < abs(B[1] - A[1])

ie = InferenceEngine(model("yolo26-n_640x640.dxnn"))  # NMS가 없는 모델: ARM 후처리가 가볍다
video = VideoSource(a.video, (W, H))
out = VideoWriter(a.save, (W, H), video.fps) if a.save else None
tracker = Tracker()
counted = {}  # 트랙 ID → 이미 센 방향
total = {}  # (분류, 방향) → 수
per_min = {}  # (분, 분류, 방향) → 수


def handle(frame, dets, idx):
    t_sec = idx / video.fps
    for tr in tracker.update(dets):
        if len(tr.path) >= 2 and tr.id not in counted:
            s0, s1 = side(tr.path[-2], A, B), side(tr.path[-1], A, B)
            if s0 != 0 and s1 != 0 and s0 != s1:  # 선을 넘었다
                # 처음 본 곳 → 지금 위치로 방향을 정한다
                (px, py), (qx, qy) = tr.path[0], tr.path[-1]
                if vertical:
                    d = "right" if qx > px else "left"
                else:
                    d = "down" if qy > py else "up"
                counted[tr.id] = d
                key = (COCO[tr.cls], d)
                total[key] = total.get(key, 0) + 1
                k2 = (int(t_sec // 60), *key)
                per_min[k2] = per_min.get(k2, 0) + 1
        bx = tr.box.astype(int)
        c = color(tr.cls)
        cv2.rectangle(frame, tuple(bx[:2]), tuple(bx[2:]), c, 2)
        put_text(frame, f"{COCO[tr.cls]} #{tr.id}", (bx[0] + 4, bx[1] + 18), 0.5, bg=c)
    if out:
        cv2.line(frame, tuple(map(int, A)), tuple(map(int, B)), (0, 230, 255), 3)
        y = 36
        for (name, d), n in sorted(total.items()):
            put_text(frame, f"{name} {d} {n}", (16, y), 0.8)
            y += 36
        out.write(frame)


jobs = deque()
n = 0
t0 = time.perf_counter()
for frame in video:
    x, r, off = letterbox(frame, 640)
    jobs.append((ie.run_async([to_npu(x)]), frame.copy(), r, off, n))
    n += 1
    if len(jobs) >= 4:  # 결과는 들어온 순서대로 처리(추적은 순서가 중요)
        jid, f, r, off, i = jobs.popleft()
        handle(f, yolo26_decode(ie.wait(jid)[0], r, off, a.conf, want), i)
while jobs:
    jid, f, r, off, i = jobs.popleft()
    handle(f, yolo26_decode(ie.wait(jid)[0], r, off, a.conf, want), i)
wall = time.perf_counter() - t0
video.close()
if out:
    out.close()

with open(a.csv, "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["minute", "class", "direction", "count"])
    for (m, c, d), k in sorted(per_min.items()):
        w.writerow([m, c, d, k])
print(f"{n}프레임({n / video.fps:.0f}초 분량)을 {wall:.1f}초에 처리: {n / wall:.1f} FPS")
for (c, d), k in sorted(total.items()):
    print(f"  {c:<8} {d:<6} {k}")
print("집계:", a.csv, "| 영상:", a.save or "(저장 안 함)")
