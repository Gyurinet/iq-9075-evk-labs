# 두 NPU로 안전 장비 점검: DX-M1이 사람과 얼굴을 찾고(YOLO26n, YOLOv5s-face),
# IQ-9075 내장 NPU가 사람마다 안전모·조끼를 확인한다(GearGuardNet). ARM은 자르기·판정·기록
#   python3 apps/ppe_two_npus.py samples/ppe -o ppe_out      # 사진 폴더 → 결과 사진, report.csv
#   python3 apps/ppe_two_npus.py <영상> --save ppe.mp4       # 영상: 장마다 판정, 처리 속도
#   필요: ~/npu-venv에 ai-edge-litert, 내장 NPU용 Qualcomm 패키지와 모델(hexagon/get-models.sh)
import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hexagon"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import (
    VideoSource,
    VideoWriter,
    letterbox,
    model,
    put_text,
    to_npu,
    unletter,
    yolo26_decode,
)
from qnpu import Pool, blur, ppe, resize_pad, to_rgb
from dx_engine import InferenceEngine

ap = argparse.ArgumentParser()
ap.add_argument("src", help="사진 폴더, 사진 한 장, 또는 영상")
ap.add_argument("-o", default="ppe_out", help="사진 결과 폴더")
ap.add_argument("--save", help="영상 결과(mp4)")
ap.add_argument("--conf", type=float, default=0.45, help="안전 장비로 볼 점수")
ap.add_argument("--min-height", type=int, default=80, help="이보다 작은 사람은 판정하지 않는다")
a = ap.parse_args()

PERSON = InferenceEngine(model("yolo26-n_640x640.dxnn"))  # DX-M1
FACE = InferenceEngine(model("yolov5-s-face_640x640.dxnn"))  # DX-M1
GEAR = Pool("ppe", 2)  # 내장 NPU: HTP 0·1에 하나씩


def faces_of(pred, r, off, conf=0.3):
    """YOLOv5-face 출력 [25200, 16] → 얼굴 상자(apps/privacy_blur.py와 같은 풀이)"""
    p = pred.reshape(-1, 16)
    score = p[:, 4] * p[:, 15]
    p, score = p[score >= conf], score[score >= conf]
    if not len(p):
        return np.zeros((0, 4))
    xywh = p[:, :4].copy()
    xywh[:, :2] -= xywh[:, 2:] / 2
    idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), score.tolist(), conf, 0.4), int).reshape(-1)
    b = xywh[idx]
    b[:, 2:] += b[:, :2]
    return unletter(b, r, off)


def gear(m, crop):
    return ppe(m.run(to_rgb(resize_pad(crop, *m.size)[0])))


def ask_dx(img):
    """DX-M1에 사람·얼굴 검출을 맡기고(비동기) 바로 돌아온다"""
    x, r, off = letterbox(img, 640)
    x = to_npu(x)
    return PERSON.run_async([x]), FACE.run_async([x]), r, off


def finish(img, job):
    """DX-M1 결과를 받아 사람마다 잘라 내장 NPU에 안전 장비를 묻는다(동시에)"""
    pj, fj, r, off = job
    persons = yolo26_decode(PERSON.wait(pj)[0], r, off, 0.35, [0])
    faces = faces_of(FACE.wait(fj)[0], r, off)
    H, W = img.shape[:2]
    asks = []
    for x1, y1, x2, y2, *_ in persons:
        w, h = x2 - x1, y2 - y1
        if h < a.min_height:
            continue
        cx1, cy1 = int(max(0, x1 - w * 0.1)), int(max(0, y1 - h * 0.1))
        cx2, cy2 = int(min(W, x2 + w * 0.1)), int(min(H, y2 + h * 0.05))
        asks.append((np.array([x1, y1, x2, y2]), GEAR.submit(gear, img[cy1:cy2, cx1:cx2])))
    return faces, [(box, f.result()) for box, f in asks]


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
    files = sorted(src.glob("*.jpg")) if src.is_dir() else [src]
    out_dir = Path(a.o)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        img = cv2.imread(str(f))
        faces, result = finish(img, ask_dx(img))
        bad = sum(not (p["helmet"] >= a.conf and p["vest"] >= a.conf) for _, p in result)
        rows.append([f.name, len(result), len(result) - bad, bad])
        cv2.imwrite(str(out_dir / f.name), draw(img, faces, result))
        print(f"{f.name:<14} 사람 {len(result)}명: 착용 {len(result) - bad}, 미착용 {bad}")
    with open(out_dir / "report.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["file", "people", "ok", "missing"])
        w.writerows(rows)
    print(f"→ {out_dir}/ (report.csv)")
    sys.exit()

# 영상: DX-M1에 다음 프레임들을 미리 맡겨 두고(최대 4장), 앞 프레임의 안전 장비를 확인한다
W, H = 1280, 720
video = VideoSource(a.src, (W, H))
out = VideoWriter(a.save, (W, H), video.fps) if a.save else None
jobs = deque()
n = people = 0
t0 = time.perf_counter()
for frame in video:
    jobs.append((frame, ask_dx(frame)))
    if len(jobs) >= 4:
        f, job = jobs.popleft()
        faces, result = finish(f, job)
        people += len(result)
        n += 1
        if out:
            out.write(draw(f, faces, result))
while jobs:
    f, job = jobs.popleft()
    faces, result = finish(f, job)
    people += len(result)
    n += 1
    if out:
        out.write(draw(f, faces, result))
wall = time.perf_counter() - t0
video.close()
GEAR.close()
if out:
    out.close()
print(f"{n}프레임 {wall:.1f}초: {n / wall:.1f} FPS, 판정한 사람 {people / n:.1f}명/장")
