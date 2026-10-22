# 실습 9: NPU 코어 3개를 나눠 쓴다. 무거운 작업(분할)이 NPU를 가득 쓰는 동안,
#         급한 작업(분류)의 응답 시간을 코어 배정으로 지키는 방법을 본다
#   python3 labs/09_npu_cores.py [초]
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import model
from dx_engine import InferenceEngine, InferenceOption

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 10
B = InferenceOption.BOUND_OPTION


def load(name, bound):
    o = InferenceOption()
    o.set_bound_option(bound)                             # 이 모델이 쓸 NPU 코어
    return InferenceEngine(model(name), o)


def rand_input(ie):
    return np.random.randint(0, 255, ie.get_input_tensors_info()[0]["shape"][1:], np.uint8)


def flood(ie, stop, out):
    """무거운 작업: 요청을 계속 겹쳐 보내 NPU를 가득 채운다"""
    x, jobs, n = rand_input(ie), [], 0
    t0 = time.perf_counter()
    while not stop.is_set():
        jobs.append(ie.run_async([x]))
        if len(jobs) >= 8:
            ie.wait(jobs.pop(0))
            n += 1
    for j in jobs:
        ie.wait(j)
    out["fps"] = n / (time.perf_counter() - t0)


def urgent(ie):
    """급한 작업: 0.05초마다 한 장씩 분류하고 응답 시간을 잰다"""
    x, lat = rand_input(ie), []
    t_end = time.perf_counter() + SECONDS
    while time.perf_counter() < t_end:
        t = time.perf_counter()
        ie.run([x])
        lat.append((time.perf_counter() - t) * 1000)
        time.sleep(0.05)
    lat.sort()
    return np.median(lat), lat[int(len(lat) * 0.95) - 1]


def trial(label, cls_bound, seg_bound):
    cls = load("resnet50_224x224.dxnn", cls_bound)
    out, stop = {}, threading.Event()
    th = None
    if seg_bound is not None:
        seg = load("deeplabv3plus_mobilenetv1_512x512.dxnn", seg_bound)
        th = threading.Thread(target=flood, args=(seg, stop, out))
        th.start()
        time.sleep(1)
    med, p95 = urgent(cls)
    stop.set()
    if th:
        th.join()
    seg_txt = f"분할 {out['fps']:6.1f} FPS" if th else "분할 없음"
    print(f"{label:<30} 분류 응답 중앙값 {med:6.2f} ms, 95% {p95:6.2f} ms | {seg_txt}")


trial("분류만(기준)", B.NPU_ALL, None)
trial("함께, 코어 0·1·2 함께 쓰기", B.NPU_ALL, B.NPU_ALL)
trial("함께, 분류 코어 0 / 분할 코어 1·2", B.NPU_0, B.NPU_12)
