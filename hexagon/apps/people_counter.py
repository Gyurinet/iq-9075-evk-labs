# 통행 인원 세기: 사람을 찾아(NPU) 추적하고, 발 위치가 가상의 선을 넘으면 방향별로 센다(ARM)
#   python3 hexagon/apps/people_counter.py samples/videos/pedestrians.mp4 --save counted.mp4
#   --line x1,y1,x2,y2 : 선의 양 끝(화면 폭·높이에 대한 비율 0~1). 고정된 카메라 영상에서 쓴다
#   결과: 표시 영상(--save), 1분 단위 집계(--csv, 기본 people.csv). 얼굴은 흐리게 가린다
import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 저장소 맨 위(labkit)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hexagon/(qnpu)
from labkit import Tracker, VideoSource, VideoWriter, put_text, side
from qnpu import Pool, blur, people, resize_pad, to_rgb

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--line", default="0.5,0.3,0.5,1")
ap.add_argument("--save")
ap.add_argument("--csv", default="people.csv")
ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()

W, H = 1280, 720
x1, y1, x2, y2 = (float(v) for v in a.line.split(","))
A, B = (x1 * W, y1 * H), (x2 * W, y2 * H)
vertical = abs(B[0] - A[0]) < abs(B[1] - A[1])


def find(m, frame):
    x, r, off = resize_pad(frame, *m.size)
    faces, persons, _ = people(m.run(to_rgb(x)), r, off)
    return faces, np.column_stack([persons, np.zeros(len(persons))])  # 분류 0(사람)


pool = Pool("people", a.workers)
video = VideoSource(a.video, (W, H))
out = VideoWriter(a.save, (W, H), video.fps) if a.save else None
tracker = Tracker()
counted, total, per_min = {}, {}, {}


def handle(frame, faces, dets, idx):
    for tr in tracker.update(dets):
        if len(tr.path) >= 2 and tr.id not in counted:
            s0, s1 = side(tr.path[-2], A, B), side(tr.path[-1], A, B)
            if s0 != 0 and s1 != 0 and s0 != s1:  # 발 위치가 선을 넘었다
                (px, py), (qx, qy) = tr.path[0], tr.path[-1]
                d = (
                    ("right" if qx > px else "left")
                    if vertical
                    else ("down" if qy > py else "up")
                )
                counted[tr.id] = d
                total[d] = total.get(d, 0) + 1
                k = (int(idx / video.fps // 60), d)
                per_min[k] = per_min.get(k, 0) + 1
        if out:
            bx = tr.box.astype(int)
            c = (60, 200, 60) if tr.id in counted else (200, 160, 40)
            cv2.rectangle(frame, tuple(bx[:2]), tuple(bx[2:]), c, 2)
            put_text(frame, f"#{tr.id}", (bx[0] + 4, bx[1] + 18), 0.5, bg=c)
    if out:
        blur(frame, faces)
        cv2.line(frame, tuple(map(int, A)), tuple(map(int, B)), (0, 230, 255), 3)
        for k, (d, c) in enumerate(sorted(total.items())):
            put_text(frame, f"{d} {c}", (16, 36 + 36 * k), 0.8)
        out.write(frame)


jobs = deque()
n = 0
t0 = time.perf_counter()
for frame in video:
    jobs.append((frame, pool.submit(find, frame), n))
    n += 1
    if len(jobs) >= a.workers * 2:  # 추적은 순서가 중요하므로 들어온 순서대로 꺼낸다
        f, fut, i = jobs.popleft()
        handle(f, *fut.result(), i)
while jobs:
    f, fut, i = jobs.popleft()
    handle(f, *fut.result(), i)
wall = time.perf_counter() - t0
video.close()
pool.close()
if out:
    out.close()
with open(a.csv, "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["minute", "direction", "count"])
    for (m, d), k in sorted(per_min.items()):
        w.writerow([m, d, k])
print(f"{n}프레임({n / video.fps:.0f}초 분량)을 {wall:.1f}초에 처리: {n / wall:.1f} FPS")
for d, k in sorted(total.items()):
    print(f"  {d:<6} {k}명")
print("집계:", a.csv, "| 영상:", a.save or "(저장 안 함)")
