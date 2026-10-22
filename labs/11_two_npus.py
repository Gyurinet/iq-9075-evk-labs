# 실습 11: NPU 두 개를 함께 쓴다. DX-M1(YOLO26n)과 IQ-9075 내장 NPU(YOLOX, HTP 2개)를
#         따로 돌릴 때와 동시에 돌릴 때의 처리량을 비교한다
#   python3 labs/11_two_npus.py [초]
#   필요: ~/npu-venv에 ai-edge-litert, 내장 NPU용 Qualcomm 패키지와 모델(hexagon/get-models.sh)
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hexagon"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import cpu_seconds, letterbox, model, to_npu
from qnpu import Model, resize_pad, to_rgb
from dx_engine import InferenceEngine

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 10
img = cv2.imread(str(Path(__file__).resolve().parents[1] / "samples/images/crossing.jpg"))
dx = InferenceEngine(model("yolo26-n_640x640.dxnn"))
dx_in = to_npu(letterbox(img, 640)[0])
htp = [Model("yolox", htp=k, perf="burst") for k in (0, 1)]
htp_in = to_rgb(resize_pad(img, 640, 640)[0])


def run_dx(stop, count):
    """DX-M1: 요청을 8개까지 겹쳐 보낸다(NPU 코어 3개를 다 쓰게)"""
    jobs = deque()
    while not stop.is_set():
        jobs.append(dx.run_async([dx_in]))
        if len(jobs) >= 8:
            dx.wait(jobs.popleft())
            count[0] += 1
    while jobs:
        dx.wait(jobs.popleft())


def run_htp(stop, count, k):
    """내장 NPU: HTP마다 스레드 하나"""
    while not stop.is_set():
        htp[k].run(htp_in)
        count[0] += 1


def measure(use_dx, use_htp):
    stop = threading.Event()
    c_dx, c_h0, c_h1 = [0], [0], [0]
    ts = []
    if use_dx:
        ts.append(threading.Thread(target=run_dx, args=(stop, c_dx)))
    if use_htp:
        ts += [
            threading.Thread(target=run_htp, args=(stop, c, k))
            for k, c in ((0, c_h0), (1, c_h1))
        ]
    for t in ts:
        t.start()
    time.sleep(1)  # 준비
    c_dx[0] = c_h0[0] = c_h1[0] = 0
    c0, t0 = cpu_seconds(), time.perf_counter()
    time.sleep(SECONDS)
    wall = time.perf_counter() - t0
    n_dx, n_htp = c_dx[0], c_h0[0] + c_h1[0]
    cores = (cpu_seconds() - c0) / (time.perf_counter() - t0)
    stop.set()
    for t in ts:
        t.join()
    return n_dx / wall, n_htp / wall, cores


print(f"사진 1장을 계속 검출, {SECONDS:g}초씩")
print(f"{'':<18}{'DX-M1(YOLO26n)':>16}{'내장 NPU(YOLOX)':>18}{'합계':>9}{'CPU':>9}")
for label, a, b in (
    ("DX-M1만", True, False),
    ("내장 NPU만", False, True),
    ("둘 다 동시에", True, True),
):
    d, h, cores = measure(a, b)
    pad = 18 - sum(2 if ord(c) > 0x1100 else 1 for c in label) + len(label)
    print(f"{label:<{pad}}{d:>12.1f} FPS{h:>14.1f} FPS{d + h:>9.1f}{cores:>7.1f}코어")
