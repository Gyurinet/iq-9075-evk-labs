# 내장 NPU 모델 성능 표: 모델마다 ARM CPU(8코어)와 NPU(기본·최고 성능 모드)의
# 1장 처리 시간을 잰다
#   python3 hexagon/bench.py            # 모든 모델(5분 정도)
#   python3 hexagon/bench.py yolox ffnet
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qnpu import FILES, Model

names = sys.argv[1:] or [n for n in FILES if n != "resnet50-float"]


def avg_ms(m, seconds=3.0, at_least=5):
    """입력을 무작위로 채우고 정해진 시간 동안 반복해 평균(ms)"""
    xs = []
    for d in m.inputs:
        shape = d["shape"][1:]
        if d["dtype"] == np.float32:
            xs.append(np.random.rand(*shape).astype(np.float32))
        else:
            xs.append(np.random.randint(0, 255, shape).astype(d["dtype"]))
    m.run(*xs)
    n, t0 = 0, time.perf_counter()
    while n < at_least or time.perf_counter() - t0 < seconds:
        m.run(*xs)
        n += 1
    return (time.perf_counter() - t0) / n * 1000


def rjust(text, width):
    """한글처럼 두 칸을 차지하는 글자를 세어 오른쪽 정렬"""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return " " * max(0, width - w) + text


head = [("입력", 12), ("형식", 6), ("CPU 8코어", 11), ("NPU", 9), ("NPU 최고", 10), ("배율", 8)]
print("모델" + " " * 10 + "".join(rjust(t, w) for t, w in head))
for name in names:
    cpu = Model(name, device="cpu", threads=8)
    dims = [int(v) for v in cpu.inputs[0]["shape"][1:]]
    dims = [d for d in (dims[:-1] if dims[-1] in (1, 3) else dims) if d != 1]
    shape = "×".join(map(str, dims))
    kind = "fp16" if cpu.inputs[0]["dtype"] == np.float32 else "int8"
    t_cpu = avg_ms(cpu, seconds=2, at_least=3)
    try:
        t_npu = avg_ms(Model(name))
        t_burst = avg_ms(Model(name, perf="burst"))
    except RuntimeError:  # NPU 델리게이트가 맡지 못하는 모델(예: 반복문이 있는 모델)
        print(f"{name:<14}{shape:>12}{kind:>6}{t_cpu:>9.1f}ms   NPU에서 실행 못 함", flush=True)
        continue
    print(
        f"{name:<14}{shape:>12}{kind:>6}{t_cpu:>9.1f}ms{t_npu:>7.1f}ms"
        f"{t_burst:>8.1f}ms{t_cpu / t_burst:>6.0f}배",
        flush=True,
    )
