# 오래 돌리기 시험: HTP 2개를 최고 성능 모드로 계속 쓰면서 처리량과 온도가 어떻게 바뀌는지 본다
#   python3 hexagon/soak.py                 # YOLOX, 5분, 30초마다 기록
#   python3 hexagon/soak.py --model resnet50 --minutes 10
import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qnpu import Model

TEMPS = Path(__file__).resolve().parent / "temps.sh"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolox")
ap.add_argument("--minutes", type=float, default=5)
ap.add_argument("--every", type=float, default=30, help="기록 간격(초)")
a = ap.parse_args()

models = [Model(a.model, htp=k, perf="burst") for k in (0, 1)]
x = np.random.randint(0, 255, models[0].inputs[0]["shape"][1:]).astype(np.uint8)
count = [0, 0]
stop = threading.Event()  # 기록을 마치면 멈춘다


def work(k):
    while not stop.is_set():
        models[k].run(x)
        count[k] += 1


threads = [threading.Thread(target=work, args=(k,)) for k in (0, 1)]
for t in threads:
    t.start()
t0 = last_t = time.perf_counter()
last_n = 0
for _ in range(int(a.minutes * 60 // a.every)):
    time.sleep(a.every)
    now, n = time.perf_counter(), sum(count)
    temps = subprocess.run(["bash", str(TEMPS)], capture_output=True, text=True).stdout.strip()
    print(f"{now - t0:5.0f}초  {(n - last_n) / (now - last_t):7.1f} FPS  {temps}", flush=True)
    last_t, last_n = now, n
stop.set()
for t in threads:
    t.join()
