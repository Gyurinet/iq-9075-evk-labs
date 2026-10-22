# 구역 감시: 정해 둔 구역(작업 구역, 출입문 앞 등)에 사람이 들어오거나 오래 머물면 기록한다
#   python3 hexagon/apps/zone_guard.py samples/videos/pedestrians.mp4 --save zone.mp4
#   --zone x1,y1,x2,y2,… : 구역 꼭짓점(화면 폭·높이에 대한 비율 0~1, 시계 방향)
#   --dwell 초 : 이보다 오래 머물면 '체류'로 기록(기본 3초)
#   --min-stay 초 : 이만큼 머물러야 '진입'으로 본다(기본 0.3초, 스치기·잘못 검출 빼기)
#   사람의 위치는 발목 점(보이면)으로, 안 보이면 상자 아래 가운데로 정한다
#   결과: 사건 기록 events.csv, 들어온 순간 사진 snapshots/(얼굴은 가림), 표시 영상(--save)
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
from labkit import Tracker, VideoSource, VideoWriter, put_text
from qnpu import Pool, blur, people, resize_pad, to_rgb

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--zone", default="0.42,0.62,0.78,0.62,0.86,1.0,0.34,1.0")
ap.add_argument("--dwell", type=float, default=3.0)
ap.add_argument("--min-stay", type=float, default=0.3)
ap.add_argument("--save")
ap.add_argument("--events", default="events.csv")
ap.add_argument("--snapshots", default="snapshots")
a = ap.parse_args()

W, H = 1280, 720
v = [float(t) for t in a.zone.split(",")]
zone = (np.array(v, np.float32).reshape(-1, 2) * [W, H]).astype(np.int32)
Path(a.snapshots).mkdir(exist_ok=True)
GONE = 0.5  # 이만큼 안 보이면 나간 것으로 본다(검출이 한두 장 빠져도 괜찮게)


def find(m, frame):
    """사람 상자와 발 위치: 보이는 발목 점의 평균, 없으면 상자 아래 가운데"""
    x, r, off = resize_pad(frame, *m.size)
    faces, persons, kps = people(m.run(to_rgb(x)), r, off)
    feet = []
    for (x1, _, x2, y2, _), k in zip(persons, kps, strict=True):
        ankles = k[[15, 16]]
        ankles = ankles[ankles[:, 2] >= 0.5]
        feet.append(tuple(ankles[:, :2].mean(0)) if len(ankles) else ((x1 + x2) / 2, y2))
    dets = np.column_stack([persons, np.zeros(len(persons))])  # 분류 0(사람)
    return faces, dets, feet


pool = Pool("people", 2)
video = VideoSource(a.video, (W, H))
out = VideoWriter(a.save, (W, H), video.fps) if a.save else None
tracker = Tracker()
inside = {}  # 트랙 ID → [들어온 시각, 마지막으로 본 시각, 진입 확정]
dwelled = set()
log = []  # (시각, 사건, ID, 머문 시간)


def handle(frame, faces, dets, feet, idx):
    now = idx / video.fps
    seen = set()
    blur(frame, faces)  # 기록 사진과 영상에는 얼굴을 가린다
    for tr in tracker.update(dets):
        if not len(dets):
            break
        j = int(np.argmin(np.abs(dets[:, :4] - tr.box).sum(1)))  # 이 트랙의 이번 검출
        foot = tuple(map(float, feet[j]))
        if cv2.pointPolygonTest(zone, foot, False) < 0:  # 발이 구역 밖
            continue
        if tr.id not in inside:
            inside[tr.id] = [now, now, False]
        inside[tr.id][1] = now
        stay = now - inside[tr.id][0]
        if not inside[tr.id][2]:
            if stay < a.min_stay:
                continue
            inside[tr.id][2] = True
            log.append((inside[tr.id][0], "enter", tr.id, 0.0))
            cv2.imwrite(f"{a.snapshots}/{inside[tr.id][0]:07.2f}s_id{tr.id}.jpg", frame)
        seen.add(tr.id)
        if stay >= a.dwell and tr.id not in dwelled:
            dwelled.add(tr.id)
            log.append((now, "dwell", tr.id, stay))
        c = (0, 0, 255) if stay >= a.dwell else (0, 160, 255)
        bx = tr.box.astype(int)
        cv2.rectangle(frame, tuple(bx[:2]), tuple(bx[2:]), c, 3)
        cv2.circle(frame, tuple(map(int, foot)), 7, c, -1)
        put_text(frame, f"#{tr.id} {stay:.1f}s", (bx[0] + 4, bx[1] + 22), 0.6, bg=c)
    for tid, (t_in, t_last, sure) in list(inside.items()):  # 구역에서 나감(또는 사라짐)
        if now - t_last > GONE:
            if sure:
                log.append((t_last, "leave", tid, t_last - t_in))
            del inside[tid]
    if out:
        color = (0, 0, 255) if seen else (0, 200, 0)
        layer = frame.copy()
        cv2.fillPoly(layer, [zone], color)
        frame = cv2.addWeighted(layer, 0.25, frame, 0.75, 0)
        cv2.polylines(frame, [zone], True, color, 3)
        enters = sum(1 for e in log if e[1] == "enter")
        put_text(frame, f"t={now:5.1f}s  in zone: {len(seen)}  events: {enters}", (16, 36), 0.8)
        out.write(frame)


jobs = deque()
n = 0
t0 = time.perf_counter()
for frame in video:
    jobs.append((frame, pool.submit(find, frame), n))
    n += 1
    if len(jobs) >= 4:  # 추적은 순서가 중요하므로 들어온 순서대로 꺼낸다
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

with open(a.events, "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["time_s", "event", "id", "stay_s"])
    for t, e, tid, s in sorted(log, key=lambda e: e[0]):
        w.writerow([f"{t:.2f}", e, tid, f"{s:.1f}"])
enters = [e for e in log if e[1] == "enter"]
print(f"{n}프레임을 {wall:.1f}초에 처리: {n / wall:.1f} FPS")
print(
    f"구역 진입 {len(enters)}건, {a.dwell:g}초 넘게 머묾 {len(dwelled)}건"
    f" → {a.events}, 사진 {a.snapshots}/"
)
