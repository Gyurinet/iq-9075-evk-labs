# 실습 3: 같은 모델 파일을 ARM CPU(XNNPACK)와 내장 NPU(Hexagon HTP)에서 실행해 비교한다
#   python3 hexagon/labs/h03_cpu_vs_npu.py resnet50 [이미지 …]
#   모델: resnet50, mobilenet_v2, yolox(8비트 양자화), resnet50-float(실수 → NPU에서 fp16)
import sys
import threading
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qnpu import Model, cpu_seconds, labels, resize_pad, to_rgb

name = sys.argv[1] if len(sys.argv) > 1 else "resnet50"
images = sys.argv[2:]
N = 30


def pad(text, width):
    """한글처럼 두 칸을 차지하는 글자를 세어 왼쪽 정렬"""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - w)


def prepare(m, img):
    h, w = m.size
    return to_rgb(resize_pad(img, h, w)[0])


def measure(m, x):
    """N회 실행: 평균 시간(ms)과 그동안 쓴 CPU 코어 수"""
    m.run(x)
    c0, t0 = cpu_seconds(), time.perf_counter()
    for _ in range(N):
        m.run(x)
    wall = time.perf_counter() - t0
    return wall / N * 1000, (cpu_seconds() - c0) / wall


img = cv2.imread(images[0]) if images else np.full((480, 640, 3), 128, np.uint8)
print(f"모델 {name}, {N}회 평균")
print(f"{pad('실행 방법', 26)}{'장당(ms)':>8}{'FPS':>9}{'CPU 코어':>9}")
rows = []
for threads in (1, 4, 8):
    m = Model(name, device="cpu", threads=threads)
    ms, cores = measure(m, prepare(m, img))
    rows.append((f"ARM CPU {threads}코어", ms, cores))
npu = Model(name)
ms, cores = measure(npu, prepare(npu, img))
rows.append(("NPU 1개, 기본 모드", ms, cores))
burst = Model(name, perf="burst")
ms, cores = measure(burst, prepare(burst, img))
rows.append(("NPU 1개, 최고 성능 모드", ms, cores))


def two_htps(seconds=5):
    """HTP 2개에 스레드 하나씩: 두 NPU를 함께 쓸 때의 처리량"""
    models = [Model(name, htp=k, perf="burst") for k in (0, 1)]
    x = prepare(models[0], img)
    done = [0, 0]
    stop = time.perf_counter() + seconds

    def work(k):
        while time.perf_counter() < stop:
            models[k].run(x)
            done[k] += 1

    c0, t0 = cpu_seconds(), time.perf_counter()
    ts = [threading.Thread(target=work, args=(k,)) for k in (0, 1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    return wall / sum(done) * 1000, (cpu_seconds() - c0) / wall


ms, cores = two_htps()
rows.append(("NPU 2개, 최고 성능 모드", ms, cores))
for label, ms, cores in rows:
    print(f"{pad(label, 26)}{ms:>10.2f}{1000 / ms:>9.1f}{cores:>11.2f}")
print(
    f"ARM CPU 8코어 대비: NPU 1개(기본) {rows[2][1] / rows[3][1]:.1f}배,"
    f" NPU 2개(최고 성능) {rows[2][1] / rows[5][1]:.1f}배"
)

# 분류 모델이면 CPU와 NPU의 1순위가 같은지 본다(같은 모델 파일이라도 계산 방식이 다르다)
if images and name != "yolox":
    cpu = Model(name, device="cpu", threads=8)
    out = "class_logits"
    names = labels(name)
    same = 0
    for p in images:
        im = cv2.imread(p)
        a = int(np.argmax(cpu.run(prepare(cpu, im))[out]))
        b = int(np.argmax(npu.run(prepare(npu, im))[out]))
        same += a == b
        if a != b:  # 다른 것만 보인다
            print(f"  다름: {Path(p).name} CPU {names[a]}, NPU {names[b]}")
    print(f"1순위 일치: {same}/{len(images)}")
