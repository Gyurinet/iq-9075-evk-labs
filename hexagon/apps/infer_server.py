# 사내 추론 서버: 사내망의 PC·프로그램이 사진을 보내면 내장 NPU로 분석해 JSON으로 돌려준다
#   python3 hexagon/apps/infer_server.py [--port 8080]
#   GET  /health                    서버 상태와 모델
#   POST /detect?conf=0.4           사진 → 검출한 물체 목록(COCO 80종, YOLOX)
#   POST /classify?top=5            사진 → 분류 상위 n개(ImageNet 1000종, ResNet50)
#   POST /people                    사진 → 사람·얼굴 수와 사람 상자(FootTrackNet)
#   POST /tags?labels=a,b,c         사진 → 주어진 낱말(영어) 중 맞는 순서(CLIP, 학습 없이)
# 호출: python3 apps/infer_client.py http://<장비 IP>:8080/detect <사진>  (DX-M1용과 같다)
#       curl --data-binary @<사진> http://<장비 IP>:8080/people
import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 저장소 맨 위(labkit)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hexagon/(qnpu)
from clip_tokenizer import Tokenizer
from labkit import COCO
from qnpu import Pool, labels, people, resize_pad, to_rgb, yolox

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--port", type=int, default=8080)
a = ap.parse_args()

# 모델은 서버를 켤 때 한 번만 NPU에 올린다. 모델마다 HTP 0·1에 하나씩 올려 요청을 겹쳐 처리한다
print("모델을 NPU에 올리는 중…", flush=True)
DET, CLS, PPL, CLIP = Pool("yolox"), Pool("resnet50"), Pool("people"), Pool("clip")
NAMES = labels("resnet50")
TOK = Tokenizer()
stats = {"requests": 0, "started": time.time()}
lock = threading.Lock()


def detect(m, img, conf):
    x, r, off = resize_pad(img, *m.size)
    return [
        {
            "label": COCO[int(c)],
            "score": round(float(s), 3),
            "box": [round(float(v)) for v in b],
        }
        for *b, s, c in yolox(m.run(to_rgb(x)), r, off, conf)
    ]


def classify(m, img, top):
    z = m.run(to_rgb(cv2.resize(img, m.size[::-1], interpolation=cv2.INTER_AREA)))
    z = z["class_logits"]
    p = np.exp(z - z.max())
    p /= p.sum()
    return [
        {"label": NAMES[i], "score": round(float(p[i]), 4)} for i in np.argsort(p)[::-1][:top]
    ]


def count_people(m, img):
    x, r, off = resize_pad(img, *m.size)
    faces, persons, _ = people(m.run(to_rgb(x)), r, off)
    boxes = [
        {"label": "person", "score": round(float(s), 3), "box": [round(float(v)) for v in b]}
        for *b, s in persons
    ]
    return {"people": len(persons), "faces": len(faces), "objects": boxes}


def tags(m, img, words):
    """CLIP: 사진 벡터와 "a photo of a <낱말>" 글 벡터를 비교(낱말 5개까지)"""
    h, w = img.shape[:2]
    s = 224 / min(h, w)
    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    y, x = (h - 224) // 2, (w - 224) // 2
    rgb = to_rgb(img[y : y + 224, x : x + 224]).astype(np.float32) / 255
    words = words[:5]
    text = TOK([f"a photo of a {t}" for t in words] + [""] * (5 - len(words)))
    out = m.run(rgb, text)
    iv = out["image_features"] / np.linalg.norm(out["image_features"])
    tv = out["text_features"][: len(words)]
    sim = tv @ iv / np.linalg.norm(tv, axis=1)
    p = np.exp(100 * (sim - sim.max()))
    p /= p.sum()
    return [{"label": words[i], "score": round(float(p[i]), 4)} for i in np.argsort(p)[::-1]]


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            return self.reply(
                200,
                {
                    "ok": True,
                    "models": ["yolox", "resnet50", "foot_track_net", "openai_clip"],
                    "requests": stats["requests"],
                    "uptime_s": round(time.time() - stats["started"]),
                },
            )
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        t0 = time.perf_counter()
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        size = int(self.headers.get("Content-Length", 0))
        if not 0 < size <= 20_000_000:
            return self.reply(400, {"error": "사진 파일 내용을 본문으로 보내세요(20 MB 이하)"})
        img = cv2.imdecode(np.frombuffer(self.rfile.read(size), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return self.reply(400, {"error": "사진을 읽을 수 없습니다(JPEG·PNG)"})
        if url.path == "/detect":
            result = {"objects": DET.submit(detect, img, float(q.get("conf", 0.4))).result()}
        elif url.path == "/classify":
            result = {"top": CLS.submit(classify, img, int(q.get("top", 5))).result()}
        elif url.path == "/people":
            result = PPL.submit(count_people, img).result()
        elif url.path == "/tags" and q.get("labels"):
            words = [w.strip() for w in q["labels"].split(",") if w.strip()]
            result = {"top": CLIP.submit(tags, img, words).result()}
        else:
            return self.reply(404, {"error": "not found"})
        with lock:
            stats["requests"] += 1
        result.update(
            width=img.shape[1],
            height=img.shape[0],
            ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        self.reply(200, result)

    def log_message(self, fmt, *args):  # 요청마다 한 줄씩 남긴다
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")


ThreadingHTTPServer.request_queue_size = 128  # 동시에 몰리는 연결을 받아 둘 줄(기본 5면 끊긴다)
ThreadingHTTPServer.daemon_threads = True
server = ThreadingHTTPServer((a.host, a.port), Handler)
print(f"추론 서버 시작: http://{a.host}:{a.port}  (끝내려면 Ctrl+C)", flush=True)
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
