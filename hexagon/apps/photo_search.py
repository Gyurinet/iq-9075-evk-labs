# 사진 검색: CLIP으로 사진마다 특징 벡터를 뽑아 두고(NPU), 영어 문장이나 사진으로 찾는다
#   python3 hexagon/apps/photo_search.py index samples/photos       # 폴더 색인(photos.npz)
#   python3 hexagon/apps/photo_search.py text "a yellow forklift"   # 문장으로 찾기
#   python3 hexagon/apps/photo_search.py image <사진>               # 비슷한 사진 찾기
# 이 CLIP 모델은 사진 1장과 문장 5개를 함께 받아 두 벡터를 한 번에 낸다.
# 한쪽만 필요하면 다른 쪽은 빈 값으로 채운다
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clip_tokenizer import Tokenizer
from qnpu import Model

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["index", "text", "image"])
ap.add_argument("arg", help="index: 사진 폴더, text: 영어 문장, image: 사진")
ap.add_argument("--db", default="photos.npz")
ap.add_argument("--top", type=int, default=5)
a = ap.parse_args()

m = Model("clip")
tok = Tokenizer()
EMPTY_TEXT = tok([""] * 5)
EMPTY_IMAGE = np.zeros((224, 224, 3), np.float32)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def image_vec(path):
    """사진 → 512차원 벡터. 짧은 변을 224로 줄이고 가운데를 자른다(정규화는 모델 안에서)"""
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    s = 224 / min(h, w)
    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    y, x = (h - 224) // 2, (w - 224) // 2
    rgb = cv2.cvtColor(img[y : y + 224, x : x + 224], cv2.COLOR_BGR2RGB)
    return unit(m.run(rgb.astype(np.float32) / 255, EMPTY_TEXT)["image_features"])


def text_vec(sentence):
    t = tok([sentence] + [""] * 4)
    return unit(m.run(EMPTY_IMAGE, t)["text_features"][0])


if a.cmd == "index":
    files = sorted(
        p for p in Path(a.arg).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    t0 = time.perf_counter()
    vecs = np.stack([image_vec(p) for p in files])
    wall = time.perf_counter() - t0
    np.savez(a.db, vecs=vecs, files=np.array([str(p) for p in files]))
    print(
        f"{len(files)}장 색인: {wall:.1f}초({len(files) / wall:.1f}장/초,"
        f" NPU {m.last_ms:.0f} ms/장) → {a.db}"
    )
    sys.exit()

db = np.load(a.db)
q = text_vec(a.arg) if a.cmd == "text" else image_vec(a.arg)
sim = db["vecs"] @ q  # 코사인 유사도(길이 1인 벡터끼리 내적)
print(f"질의: {a.arg}")
for i in np.argsort(sim)[::-1][: a.top]:
    print(f"  {sim[i]:.3f}  {Path(str(db['files'][i])).name}")
