# 사진 검색: 사진마다 특징 벡터(CLIP 이미지 임베딩)를 NPU로 뽑아 두고,
# 비슷한 사진·중복 사진을 찾는다
#   python3 apps/photo_search.py index samples/photos        # 폴더의 사진을 색인(photos.npz)
#   python3 apps/photo_search.py query samples/images/forklift.jpg # 비슷한 사진 5장
#   python3 apps/photo_search.py dups                             # 거의 같은 사진 묶음
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import model
from dx_engine import InferenceEngine

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["index", "query", "dups"])
ap.add_argument("path", nargs="?")
ap.add_argument("--db", default="photos.npz")
ap.add_argument("--top", type=int, default=5)
ap.add_argument("--th", type=float, default=0.9, help="중복으로 볼 유사도(0~1)")
a = ap.parse_args()

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
ie = None


def embed(path):
    """사진 → 길이 1인 512차원 벡터.
    짧은 변을 256으로 줄이고 가운데를 256×256으로 자른다(CLIP 전처리)"""
    global ie
    ie = ie or InferenceEngine(model("clip-img_vit-b32_256x256_datacomp-s34b-b86k.dxnn"))
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    s = 256 / min(h, w)
    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    y, x = (h - 256) // 2, (w - 256) // 2
    x_in = (
        cv2.cvtColor(img[y : y + 256, x : x + 256], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    )
    x_in = np.ascontiguousarray(((x_in - MEAN) / STD).transpose(2, 0, 1)[None], np.float32)
    v = ie.run([x_in])[0].reshape(-1)
    return v / np.linalg.norm(v)


if a.cmd == "index":
    files = sorted(
        p for p in Path(a.path).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    t0 = time.perf_counter()
    vecs = np.stack([embed(p) for p in files])
    wall = time.perf_counter() - t0
    np.savez(a.db, vecs=vecs, files=np.array([str(p) for p in files]))
    print(f"{len(files)}장 색인: {wall:.1f}초({len(files) / wall:.1f}장/초) → {a.db}")
    sys.exit()

db = np.load(a.db)
vecs, files = db["vecs"], db["files"]
if a.cmd == "query":
    sim = vecs @ embed(a.path)  # 길이 1인 벡터끼리의 내적 = 코사인 유사도
    for i in np.argsort(sim)[::-1][: a.top]:
        print(f"  {sim[i]:.3f}  {files[i]}")
else:
    sim = vecs @ vecs.T
    shown = set()
    for i in range(len(files)):
        same = [j for j in np.where(sim[i] >= a.th)[0] if j != i and j not in shown]
        if same and i not in shown:
            shown.update([i, *same])
            print(
                "  "
                + " = ".join(
                    Path(files[k]).name + (f"({sim[i, k]:.3f})" if k != i else "")
                    for k in [i, *same]
                )
            )
    if not shown:
        print("  중복으로 보이는 사진이 없습니다.")
