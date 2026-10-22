# 사내 추론 서버: 사내망의 PC·프로그램이 사진을 보내면 NPU로 분석해 JSON으로 돌려준다
#   python3 apps/infer_server.py [--port 8080]
#   GET  /health                 서버 상태와 모델
#   POST /detect?conf=0.3        사진(JPEG·PNG 파일 내용) → 검출한 물체 목록(COCO 80종)
#   POST /classify?top=5         사진 → 분류 상위 n개(ImageNet 1000종)
# 예) curl --data-binary @samples/images/crossing.jpg http://<장비 IP>:8080/detect
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import COCO, imagenet_labels, letterbox, model, to_npu, yolo26_decode
from dx_engine import InferenceEngine

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--port", type=int, default=8080)
a = ap.parse_args()

# 모델은 서버를 켤 때 한 번만 NPU에 올리고, 여러 요청이 같이 쓴다(요청마다 올리면 느리다)
DET = InferenceEngine(model("yolo26-n_640x640.dxnn"))
CLS = InferenceEngine(model("resnet50_224x224.dxnn"))
LABELS = imagenet_labels()
stats = {"requests": 0, "started": time.time()}
lock = threading.Lock()


def detect(img, conf):
    x, r, off = letterbox(img, 640)
    dets = yolo26_decode(DET.run([to_npu(x)])[0], r, off, conf)
    return [
        {
            "label": COCO[int(c)],
            "score": round(float(s), 3),
            "box": [round(float(v)) for v in (x1, y1, x2, y2)],
        }
        for x1, y1, x2, y2, s, c in dets
    ]


def classify(img, top):
    x = to_npu(cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA))
    z = CLS.run([x])[0].reshape(-1)
    p = np.exp(z - z.max())
    p /= p.sum()
    return [
        {"label": LABELS[i], "score": round(float(p[i]), 4)} for i in np.argsort(p)[::-1][:top]
    ]


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
                    "models": ["yolo26n (detect)", "resnet50 (classify)"],
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
            result = {"objects": detect(img, float(q.get("conf", 0.3)))}
        elif url.path == "/classify":
            result = {"top": classify(img, int(q.get("top", 5)))}
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
# 요청마다 스레드 하나: 여러 요청이 NPU를 겹쳐 쓴다
server = ThreadingHTTPServer((a.host, a.port), Handler)
print(f"추론 서버 시작: http://{a.host}:{a.port}  (끝내려면 Ctrl+C)", flush=True)
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
