# 안전 장비 점검: 사람을 찾고(NPU) 사람마다 안전모·안전 조끼를 확인한다(NPU). 두 모델을 잇는 예
#   python3 hexagon/apps/ppe_check.py samples/ppe -o ppe_out   # 사진 폴더 → 결과, report.csv
#   python3 hexagon/apps/ppe_check.py <영상> --save ppe.mp4    # 영상: 추적하며 판정, events.csv
# 초록 = 둘 다 착용, 빨강 = 하나라도 없음. 얼굴은 흐리게 가려 저장한다
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
from qnpu import Model, blur, people, ppe, resize_pad, to_rgb

ap = argparse.ArgumentParser()
ap.add_argument("src", help="사진 폴더, 사진 한 장, 또는 영상")
ap.add_argument("-o", default="ppe_out", help="사진 결과 폴더")
ap.add_argument("--save", help="영상 결과(mp4)")
ap.add_argument("--events", default="events.csv", help="영상: 미착용 기록(CSV)")
ap.add_argument("--conf", type=float, default=0.45, help="안전 장비로 볼 점수")
ap.add_argument(
    "--min-height", type=int, default=80, help="이보다 작은(먼) 사람은 판정하지 않는다(화소)"
)
a = ap.parse_args()

finder = Model("people")  # 사람·얼굴
gear = Model("ppe")  # 안전모·조끼


def check(img):
    """사진 한 장 → 얼굴 상자, [(사람 상자, {"helmet": 점수, "vest": 점수}), ...]"""
    x, r, off = resize_pad(img, *finder.size)
    faces, persons, _ = people(finder.run(to_rgb(x)), r, off)
    result = []
    H, W = img.shape[:2]
    for x1, y1, x2, y2, _ in persons:
        w, h = x2 - x1, y2 - y1
        if h < a.min_height:
            continue
        # 사람 상자를 조금 넓혀 자른다(안전모가 상자 위로 나오는 경우)
        cx1, cy1 = int(max(0, x1 - w * 0.1)), int(max(0, y1 - h * 0.1))
        cx2, cy2 = int(min(W, x2 + w * 0.1)), int(min(H, y2 + h * 0.05))
        crop = resize_pad(img[cy1:cy2, cx1:cx2], *gear.size)[0]  # 320×192(세로로 긴 입력)
        result.append((np.array([x1, y1, x2, y2]), ppe(gear.run(to_rgb(crop)))))
    return faces, result


def draw(img, faces, result):
    blur(img, faces)
    t = max(2, round(min(img.shape[:2]) / 300))
    for box, p in result:
        ok = p["helmet"] >= a.conf and p["vest"] >= a.conf
        c = (60, 200, 60) if ok else (40, 40, 230)
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, t)
        label = ("H" if p["helmet"] >= a.conf else "-") + ("V" if p["vest"] >= a.conf else "-")
        put_text(img, label, (x1 + 4, max(20, y1 - 6)), 0.4 * t, bg=c)
    return img


src = Path(a.src)
if src.is_dir() or src.suffix.lower() in (".jpg", ".jpeg", ".png"):
    files = (
        sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if src.is_dir()
        else [src]
    )
    out_dir = Path(a.o)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    for f in files:
        img = cv2.imread(str(f))
        faces, result = check(img)
        bad = sum(not (p["helmet"] >= a.conf and p["vest"] >= a.conf) for _, p in result)
        rows.append([f.name, len(result), len(result) - bad, bad])
        cv2.imwrite(str(out_dir / f.name), draw(img, faces, result))
        print(f"{f.name:<14} 사람 {len(result)}명: 착용 {len(result) - bad}, 미착용 {bad}")
    with open(out_dir / "report.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["file", "people", "ok", "missing"])
        w.writerows(rows)
    print(f"{len(files)}장 {time.perf_counter() - t0:.1f}초 → {out_dir}/ (report.csv)")
    sys.exit()

# 영상: 사람을 추적하고, 최근 15번 판정 중 과반이 미착용이면 그 사람을 미착용으로 본다
# (한 장의 판정 실수에 흔들리지 않게)
W, H = 1280, 720
video = VideoSource(a.src, (W, H))
out = VideoWriter(a.save, (W, H), video.fps) if a.save else None
tracker = Tracker()
votes = {}  # 트랙 ID → 최근 판정(True=착용)
reported = set()
events = open(a.events, "w", newline="")
log = csv.writer(events)
log.writerow(["time_s", "track", "helmet", "vest"])
n = 0
t0 = time.perf_counter()
for frame in video:
    faces, result = check(frame)
    dets = [[*box, 1.0, 0] for box, _ in result]
    tracks = tracker.update(dets)
    shown = []
    for tr in tracks:
        # 추적 상자와 가장 가까운 이번 판정을 찾는다
        best = max(result, key=lambda bp: -np.abs(bp[0] - tr.box).sum(), default=None)
        if best is None:
            continue
        p = best[1]
        v = votes.setdefault(tr.id, deque(maxlen=15))
        v.append(p["helmet"] >= a.conf and p["vest"] >= a.conf)
        if len(v) >= 5 and sum(v) < len(v) / 2 and tr.id not in reported:
            reported.add(tr.id)
            log.writerow(
                [f"{n / video.fps:.1f}", tr.id, f"{p['helmet']:.2f}", f"{p['vest']:.2f}"]
            )
        shown.append((tr.box, p))
    if out:
        out.write(draw(frame, faces, shown))
    n += 1
wall = time.perf_counter() - t0
video.close()
events.close()
if out:
    out.close()
print(
    f"{n}프레임 {wall:.1f}초({n / wall:.1f} FPS),"
    f" 미착용으로 기록한 사람 {len(reported)}명 → {a.events}"
)
