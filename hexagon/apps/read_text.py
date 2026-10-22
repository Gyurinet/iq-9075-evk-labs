# 글자 읽기(OCR): 사진에서 글자가 있는 곳을 찾고(EasyOCR 검출기) 영문·숫자를 읽는다(인식기)
#   python3 hexagon/apps/read_text.py samples/signs/*.jpg [-o 결과폴더]
#   표지판, 설비 이름표, 상자 라벨처럼 인쇄된 영문·숫자에 알맞다(한글은 이 모델로 읽지 못한다)
#   결과: 글자 상자를 그린 사진(-o), 읽은 글자 CSV(--csv)
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 저장소 맨 위(labkit)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hexagon/(qnpu)
from labkit import put_text
from qnpu import Model, resize_pad, to_rgb

ap = argparse.ArgumentParser()
ap.add_argument("images", nargs="+")
ap.add_argument("-o", default="text_out", help="결과 사진 폴더")
ap.add_argument("--csv", default="text.csv")
ap.add_argument("--min-conf", type=float, default=0.3, help="이보다 자신 없는 글자는 버린다")
ap.add_argument("--threads", type=int, default=8, help="글자 읽기(ARM CPU)에 쓸 코어 수")
a = ap.parse_args()

# 인식기가 내는 97가지: 0번은 '글자 없음', 1번부터 아래 글자(EasyOCR 영어 모델)
CHARS = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €" + (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)
det = Model("ocr-det")  # 글자 찾기: NPU(8비트)
# 글자 읽기 모델은 반복문(WHILE)과 크기가 바뀌는 텐서가 있어 NPU가 맡지 못한다 → ARM CPU
rec = Model("ocr-rec", device="cpu", threads=a.threads)


def find_text(img):
    """검출기(CRAFT): 글자 점수와 글자 사이 연결 점수 지도 → 단어 상자(네 꼭짓점)"""
    x, r, off = resize_pad(img, *det.size)  # 608×800
    out = det.run(to_rgb(x))["results"]  # [304, 400, 2]
    text, link = out[..., 0], out[..., 1]
    char = (text > 0.4).astype(np.uint8)
    joined = np.clip(char + (link > 0.4), 0, 1).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=4)
    boxes = []
    for k in range(1, n):
        x0, y0, w, h, area = stats[k]
        if area < 10 or text[lab == k].max() < 0.7:  # 작거나 글자 점수가 약하면 뺀다
            continue
        seg = (lab == k).astype(np.uint8)
        grow = int(np.sqrt(area * min(w, h) / (w * h)) * 2)  # 글자 둘레만큼 넓힌다
        seg = cv2.dilate(seg, np.ones((1 + grow, 1 + grow), np.uint8))
        pts = cv2.findNonZero(seg).reshape(-1, 2).astype(np.float32)
        quad = cv2.boxPoints(cv2.minAreaRect(pts)) * 2  # 지도는 입력의 절반 크기
        quad = (quad - off) / r  # 원래 사진 좌표로
        boxes.append(quad)
    return reading_order(boxes)


def reading_order(boxes):
    """읽는 순서: 세로 가운데가 높이의 절반 안으로 가까운 상자끼리 한 줄, 줄 안은 왼쪽부터"""
    lines = []
    for q in sorted(boxes, key=lambda q: q[:, 1].mean()):
        cy, h = q[:, 1].mean(), np.ptp(q[:, 1])
        if lines and abs(cy - lines[-1][0]) < h / 2:
            lines[-1][1].append(q)
        else:
            lines.append([cy, [q]])
    return [q for _, line in lines for q in sorted(line, key=lambda q: q[:, 0].min())]


def read(gray, quad):
    """인식기: 단어 상자를 바로 펴서 높이 64로 맞추고 글자를 읽는다(CTC 탐욕 풀이)"""
    sm, df = quad.sum(1), quad[:, 1] - quad[:, 0]
    q = np.float32([quad[sm.argmin()], quad[df.argmin()], quad[sm.argmax()], quad[df.argmax()]])
    w = int(
        max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[3] - q[2]))
    )  # 왼위·오위·오아·왼아
    h = int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
    if w < 4 or h < 4:
        return "", 0.0
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    crop = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(q, dst), (w, h))
    nw = min(800, max(1, round(w * 64 / h)))
    line = np.full((64, 800), crop[0, 0], np.uint8)  # 남는 오른쪽은 첫 화소 색으로
    line[:, :nw] = cv2.resize(crop, (nw, 64), interpolation=cv2.INTER_AREA)
    logits = rec.run((line.astype(np.float32) / 255)[..., None])["output_preds"]  # [199, 97]
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    best = p.argmax(1)
    keep = (best != 0) & np.r_[True, best[1:] != best[:-1]]  # 같은 글자 반복과 빈칸을 지운다
    text = "".join(CHARS[i - 1] for i in best[keep])
    conf = (
        float(p.max(1)[keep].prod() ** (2 / np.sqrt(max(1, keep.sum())))) if keep.any() else 0
    )
    return text, conf


Path(a.o).mkdir(parents=True, exist_ok=True)
rows = []
for path in a.images:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    boxes = find_text(img)
    t1 = time.perf_counter()
    words = [(q, *read(gray, q)) for q in boxes]
    words = [(q, t, c) for q, t, c in words if t.strip() and c >= a.min_conf]
    t2 = time.perf_counter()
    th = max(2, round(min(img.shape[:2]) / 400))
    for q, t, c in words:
        cv2.polylines(img, [q.astype(np.int32)], True, (0, 200, 0), th)
        put_text(img, t, (int(q[:, 0].min()), max(20, int(q[:, 1].min()) - 8)), 0.35 * th)
        rows.append([Path(path).name, t, f"{c:.2f}", *q.astype(int).reshape(-1)])
    cv2.imwrite(str(Path(a.o) / Path(path).name), img)
    text = " ".join(t for _, t, _ in words)
    print(
        f"{Path(path).name[:28]:<28} 글자 상자 {len(boxes)}개, 찾기 {(t1 - t0) * 1000:.0f} ms,"
        f" 읽기 {(t2 - t1) * 1000:.0f} ms | {text}"
    )
with open(a.csv, "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["file", "text", "conf", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"])
    w.writerows(rows)
print(f"→ {a.o}/, {a.csv}")
