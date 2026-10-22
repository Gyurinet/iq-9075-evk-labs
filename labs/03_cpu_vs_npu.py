# 실습 3: 같은 모델을 ARM CPU(ONNX Runtime)와 NPU(DX-M1)에서 실행해 비교한다
#   python3 labs/03_cpu_vs_npu.py resnet50 [이미지 …]
#   모델: resnet50, mobilenetv2, yolo11n, yolo26n
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import cpu_seconds, imagenet_labels, letterbox, model, to_npu
from dx_engine import InferenceEngine

NAMES = {
    "resnet50": "resnet50_224x224",
    "mobilenetv2": "mobilenetv2_224x224",
    "yolo11n": "yolo11-n_640x640",
    "yolo26n": "yolo26-n_640x640",
}
name = sys.argv[1] if len(sys.argv) > 1 else "resnet50"
images = sys.argv[2:]
base = NAMES[name]
size = int(base.split("_")[-1].split("x")[0])
N = 30  # 반복 횟수


def cpu_input(img):
    """ONNX 원본 모델의 입력: float32, NCHW, 정규화까지 CPU가 한다"""
    if name.startswith("yolo"):
        x = letterbox(img, size)[0]
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    else:
        x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32)


def npu_input(img):
    """NPU 모델의 입력: uint8, NHWC 그대로(정규화는 모델 안에서)"""
    if name.startswith("yolo"):
        x = letterbox(img, size)[0]
    else:
        x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return to_npu(x)


def measure(run):
    """N회 실행: 평균 지연(ms), 그동안 쓴 CPU 코어 수"""
    run()  # 준비(첫 실행) 제외
    c0, t0 = cpu_seconds(), time.perf_counter()
    lat = []
    for _ in range(N):
        t = time.perf_counter()
        run()
        lat.append((time.perf_counter() - t) * 1000)
    wall = time.perf_counter() - t0
    return float(np.mean(lat)), (cpu_seconds() - c0) / wall


img = cv2.imread(images[0]) if images else np.full((480, 640, 3), 128, np.uint8)
print(f"모델 {base}, {N}회 평균")
print(f"{'실행 방법':<22}{'장당(ms)':>10}{'FPS':>9}{'CPU 코어':>10}")

rows = []
for threads in (1, 4, 8):  # ARM Cortex-A78C 코어 수를 바꿔 본다
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(
        str(Path(model(base + ".dxnn")).with_suffix(".onnx")),
        so,
        providers=["CPUExecutionProvider"],
    )
    feed = {sess.get_inputs()[0].name: cpu_input(img)}
    ms, cores = measure(lambda s=sess, f=feed: s.run(None, f))
    rows.append((f"ARM CPU {threads}코어", ms, cores))

ie = InferenceEngine(model(base + ".dxnn"))
x = npu_input(img)
ms, cores = measure(lambda: ie.run([x]))
rows.append(("NPU 한 장씩", ms, cores))


def npu_stream(total=300, window=8):
    """요청을 window개까지 겹쳐 보낸다(run_async). 한 장 처리 시간 = 전체 시간 / 장 수"""
    c0, t0 = cpu_seconds(), time.perf_counter()
    jobs = []
    for _ in range(total):
        jobs.append(ie.run_async([x]))
        if len(jobs) >= window:
            ie.wait(jobs.pop(0))
    for j in jobs:
        ie.wait(j)
    wall = time.perf_counter() - t0
    return wall / total * 1000, (cpu_seconds() - c0) / wall


npu_stream(30)  # 준비
ms, cores = npu_stream()
rows.append(("NPU 겹쳐 보내기", ms, cores))

for label, ms, cores in rows:
    print(f"{label:<22}{ms:>10.2f}{1000 / ms:>9.1f}{cores:>10.2f}")
cpu8, one, many = rows[2], rows[3], rows[4]
print(
    f"ARM CPU 8코어 대비: 한 장씩 {cpu8[1] / one[1]:.1f}배, "
    f"겹쳐 보내면 {cpu8[1] / many[1]:.1f}배 빠릅니다."
)

# 분류 모델이면 두 결과가 같은지 확인한다(NPU 모델은 8비트 정수로 양자화되어 있다)
if images and not name.startswith("yolo"):
    labels = imagenet_labels()
    so = ort.SessionOptions()
    sess = ort.InferenceSession(
        str(Path(model(base + ".dxnn")).with_suffix(".onnx")),
        so,
        providers=["CPUExecutionProvider"],
    )
    same = 0
    for p in images:
        im = cv2.imread(p)
        a = int(np.argmax(sess.run(None, {sess.get_inputs()[0].name: cpu_input(im)})[0]))
        b = int(np.argmax(ie.run([npu_input(im)])[0]))
        same += a == b
        print(f"  {Path(p).name[:40]:<40} CPU {labels[a]:<18} NPU {labels[b]}")
    print(f"1순위 일치: {same}/{len(images)}")
