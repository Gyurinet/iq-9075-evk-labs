# 구역 감시: 정해 둔 구역(작업 구역, 출입문 앞 등)에 사람이 들어오거나 오래 머물면 기록한다
#   python3 apps/zone_guard.py samples/videos/pedestrians.mp4 --save zone.mp4
#   --zone x1,y1,x2,y2,… : 구역 꼭짓점(화면 폭·높이에 대한 비율 0~1, 시계 방향)
#   --dwell 초          : 이보다 오래 머물면 '체류'로 기록(기본 3초)
#   --min-stay 초       : 이만큼 머물러야 '진입'으로 본다(기본 0.3초).
#                         구역 끝을 스치거나 잠깐 잘못 검출된 것을 뺀다
#   결과: 사건 기록 events.csv, 들어온 순간 사진 snapshots/, 화면 표시 영상(--save)
import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import (
    COCO,
    Tracker,
    VideoSource,
    VideoWriter,
    letterbox,
    model,
    put_text,
    to_npu,
    yolo26_decode,
)
from dx_engine import InferenceEngine

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--zone", default="0.42,0.62,0.78,0.62,0.86,1.0,0.34,1.0")
ap.add_argument("--classes", default="person")
ap.add_argument("--dwell", type=float, default=3.0)
ap.add_argument("--min-stay", type=float, default=0.3)
ap.add_argument("--save")
ap.add_argument("--events", default="events.csv")
ap.add_argument("--snapshots", default="snapshots")
ap.add_argument("--conf", type=float, default=0.35)
a = ap.parse_args()

W, H = 1280, 720
v = [float(t) for t in a.zone.split(",")]
zone = (np.array(v, np.float32).reshape(-1, 2) * [W, H]).astype(np.int32)
want = [COCO.index(c) for c in a.classes.split(",")]
Path(a.snapshots).mkdir(exist_ok=True)

ie = InferenceEngine(model("yolo26-n_640x640.dxnn"))
video = VideoSource(a.video, (W, H))
out = VideoWriter(a.save, (W, H), video.fps) if a.save else None
tracker = Tracker()
inside = {}  # 트랙 ID → [들어온 시각, 마지막으로 본 시각, 진입 확정](초)
GONE = 0.5  # 이만큼 안 보이면 나간 것으로 본다(검출이 한두 장 빠져도 괜찮게)
dwelled = set()
log = []  # (시각, 사건, ID, 분류, 머문 시간)


def handle(frame, dets, idx):
    now = idx / video.fps
    seen = set()
    for tr in tracker.update(dets):
        foot = tr.foot()
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
            log.append((inside[tr.id][0], "enter", tr.id, COCO[tr.cls], 0.0))
            cv2.imwrite(f"{a.snapshots}/{inside[tr.id][0]:07.2f}s_id{tr.id}.jpg", frame)
        seen.add(tr.id)
        if stay >= a.dwell and tr.id not in dwelled:
            dwelled.add(tr.id)
            log.append((now, "dwell", tr.id, COCO[tr.cls], stay))
        c = (0, 0, 255) if stay >= a.dwell else (0, 160, 255)
        bx = tr.box.astype(int)
        cv2.rectangle(frame, tuple(bx[:2]), tuple(bx[2:]), c, 3)
        put_text(frame, f"#{tr.id} {stay:.1f}s", (bx[0] + 4, bx[1] + 22), 0.6, bg=c)
    for tid, (t_in, t_last, sure) in list(inside.items()):  # 구역에서 나감(또는 사라짐)
        if now - t_last > GONE:
            if sure:
                log.append((t_last, "leave", tid, "", t_last - t_in))
            del inside[tid]
    if out:
        layer = frame.copy()
        cv2.fillPoly(layer, [zone], (0, 0, 255) if seen else (0, 200, 0))
        frame = cv2.addWeighted(layer, 0.25, frame, 0.75, 0)
        cv2.polylines(frame, [zone], True, (0, 0, 255) if seen else (0, 200, 0), 3)
        put_text(
            frame,
            f"t={now:5.1f}s  in zone: {len(seen)}  "
            f"events: {sum(1 for e in log if e[1] == 'enter')}",
            (16, 36),
            0.8,
        )
        out.write(frame)


jobs = deque()
n = 0
t0 = time.perf_counter()
for frame in video:
    x, r, off = letterbox(frame, 640)
    jobs.append((ie.run_async([to_npu(x)]), frame.copy(), r, off, n))
    n += 1
    if len(jobs) >= 4:
        jid, f, r, off, i = jobs.popleft()
        handle(f, yolo26_decode(ie.wait(jid)[0], r, off, a.conf, want), i)
while jobs:
    jid, f, r, off, i = jobs.popleft()
    handle(f, yolo26_decode(ie.wait(jid)[0], r, off, a.conf, want), i)
wall = time.perf_counter() - t0
video.close()
if out:
    out.close()

with open(a.events, "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["time_s", "event", "id", "class", "stay_s"])
    for t, e, tid, c, s in sorted(log, key=lambda e: e[0]):
        w.writerow([f"{t:.2f}", e, tid, c, f"{s:.1f}"])
enters = [e for e in log if e[1] == "enter"]
print(f"{n}프레임을 {wall:.1f}초에 처리: {n / wall:.1f} FPS")
print(
    f"구역 진입 {len(enters)}건, {a.dwell:g}초 넘게 머묾 {len(dwelled)}건 "
    f"→ {a.events}, 사진 {a.snapshots}/"
)
