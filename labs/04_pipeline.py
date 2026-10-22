# 실습 4: 영상 객체 검출의 단계별 시간을 재고, 단계를 겹쳐(파이프라인) 처리량을 높인다
#   python3 labs/04_pipeline.py samples/videos/crossing.mp4 [--model yolo26n] [--save out.mp4]
import argparse
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import (
    Stopwatch,
    VideoSource,
    VideoWriter,
    cpu_seconds,
    draw_boxes,
    letterbox,
    model,
    to_npu,
    yolo11_decode,
    yolo26_decode,
)
from dx_engine import InferenceEngine

MODELS = {"yolo11n": "yolo11-n_640x640.dxnn", "yolo26n": "yolo26-n_640x640.dxnn"}
ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--model", choices=list(MODELS), default="yolo11n")
ap.add_argument("--save", help="겹쳐 처리한 결과를 영상(mp4)으로 저장")
a = ap.parse_args()
src, kind, save = a.video, a.model, a.save
decode = yolo11_decode if kind == "yolo11n" else yolo26_decode
ie = InferenceEngine(model(MODELS[kind]))
SIZE = (1280, 720)


def pre(frame):
    x, r, off = letterbox(frame, 640)  # ARM: 640×640에 맞추기
    return to_npu(x), r, off


# ① 한 장씩 차례로: 해독 → 전처리 → NPU → 후처리 → 그리기
print(f"[1] 한 장씩 차례로 처리 ({kind})")
sw = Stopwatch()
video = VideoSource(src, SIZE, prefetch=0)  # 미리 읽지 않는다(해독도 차례로)
n = 0
c0, t0 = cpu_seconds(), time.perf_counter()
while True:
    with sw("해독"):
        frame = video.read()
    if frame is None:
        break
    with sw("전처리"):
        x, r, off = pre(frame)
    with sw("NPU 추론"):
        out = ie.run([x])
    with sw("후처리"):
        dets = decode(out[0], r, off)
    with sw("그리기"):
        draw_boxes(frame.copy(), dets)
    n += 1
wall = time.perf_counter() - t0
video.close()
sw.report(skip=5)
print(
    f"  실제 {n}장 {wall:.1f}초 → {n / wall:.1f} FPS, "
    f"CPU {(cpu_seconds() - c0) / wall:.2f}코어\n"
)

# ② 겹쳐 처리: NPU가 앞 프레임을 계산하는 동안 ARM은 다음 프레임을 준비하고 지난 결과를 정리한다
print(f"[2] 단계를 겹쳐 처리(요청 {4}개까지 동시에)")
video = VideoSource(src, SIZE)
out_video = VideoWriter(save, SIZE) if save else None
jobs = deque()
n, total = 0, 0


def finish(job):
    global total
    jid, frame, r, off = job
    dets = decode(ie.wait(jid)[0], r, off)  # NPU 결과를 받아 ARM이 후처리
    total += len(dets)
    img = draw_boxes(frame.copy(), dets)
    if out_video:
        out_video.write(img)


c0, t0 = cpu_seconds(), time.perf_counter()
for frame in video:
    x, r, off = pre(frame)
    jobs.append((ie.run_async([x]), frame, r, off))  # NPU에 맡기고 바로 다음 프레임으로
    n += 1
    if len(jobs) >= 4:
        finish(jobs.popleft())
while jobs:
    finish(jobs.popleft())
wall = time.perf_counter() - t0
video.close()
if out_video:
    out_video.close()
print(
    f"  실제 {n}장 {wall:.1f}초 → {n / wall:.1f} FPS, "
    f"CPU {(cpu_seconds() - c0) / wall:.2f}코어, 검출 {total / n:.1f}개/장"
)
if save:
    print(f"  결과 영상: {save}")
