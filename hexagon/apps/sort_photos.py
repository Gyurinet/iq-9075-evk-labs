# 학습 없이 사진 나누기: 종류 이름(영어)만 정하면 CLIP이 사진마다 가장 맞는 종류를 고른다
#   python3 hexagon/apps/sort_photos.py samples/photos --classes "forklift,tractor,hard hat"
#   --out 폴더 : 종류별 하위 폴더로 사진을 복사한다(원본은 그대로)
#   --other 0.2 : 가장 맞는 종류의 점수가 이보다 낮으면 'other'로 둔다
#   파일 이름이 종류 이름으로 시작하면(예: forklift_1.jpg) 정답으로 보고 정확도를 낸다
import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clip_tokenizer import Tokenizer
from qnpu import Model

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--classes", required=True, help="쉼표로 나눈 종류 이름(영어)")
ap.add_argument("--out", help="종류별로 복사할 폴더")
ap.add_argument("--other", type=float, default=0.2)
ap.add_argument("--csv", default="sorted.csv")
a = ap.parse_args()

classes = [c.strip() for c in a.classes.split(",")]
m = Model("clip")
tok = Tokenizer()


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ① 종류 이름을 "a photo of a …" 문장으로 바꿔 글 벡터를 만든다(한 번에 5개씩)
prompts = [f"a photo of a {c}" for c in classes]
text = []
for i in range(0, len(prompts), 5):
    chunk = prompts[i : i + 5]
    out = m.run(np.zeros((224, 224, 3), np.float32), tok(chunk + [""] * (5 - len(chunk))))
    text.append(out["text_features"][: len(chunk)])
text = unit(np.concatenate(text))


def image_vec(path):
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    s = 224 / min(h, w)
    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    y, x = (h - 224) // 2, (w - 224) // 2
    rgb = cv2.cvtColor(img[y : y + 224, x : x + 224], cv2.COLOR_BGR2RGB)
    return unit(m.run(rgb.astype(np.float32) / 255, tok([""] * 5))["image_features"])


# ② 사진마다 글 벡터와 비교해 가장 비슷한 종류를 고른다(점수는 100배 후 확률로)
files = sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in (".jpg", ".png"))
rows, right, judged = [], 0, 0
t0 = time.perf_counter()
for p in files:
    sim = text @ image_vec(p)
    prob = np.exp(100 * (sim - sim.max()))
    prob /= prob.sum()
    k = int(np.argmax(prob))
    label = classes[k] if prob[k] >= a.other else "other"
    truth = next((c for c in classes if p.stem.replace("_", " ").startswith(c)), None)
    if truth:
        judged += 1
        right += label == truth
    rows.append([p.name, label, f"{prob[k]:.2f}"])
    if a.out:
        (Path(a.out) / label).mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, Path(a.out) / label / p.name)
wall = time.perf_counter() - t0

with open(a.csv, "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["file", "class", "prob"])
    w.writerows(rows)
count = {c: sum(r[1] == c for r in rows) for c in [*classes, "other"]}
print(", ".join(f"{c} {n}" for c, n in count.items() if n))
print(f"{len(files)}장 {wall:.1f}초 → {a.csv}" + (f", {a.out}/" if a.out else ""))
if judged:
    print(f"파일 이름이 정답인 {judged}장 중 {right}장 맞음({right / judged * 100:.0f}%)")
