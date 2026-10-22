# 실습 5: 영상 객체 검출을 한 장씩 처리할 때와, NPU 2개에 나눠 겹쳐 처리할 때를 비교한다
#   python3 hexagon/labs/h05_pipeline.py samples/videos/crossing.mp4 [--save out.mp4]
#   --workers 2 : 동시에 실행할 모델 수(HTP 0·1에 번갈아)
import argparse
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 저장소 맨 위(labkit)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hexagon/(qnpu)
from labkit import Stopwatch, VideoSource, VideoWriter, cpu_seconds, draw_boxes
from qnpu import Model, Pool, resize_pad, to_rgb, yolox

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--save", help="겹쳐 처리한 결과를 영상(mp4)으로 저장")
ap.add_argument(
    "--workers", type=int, default=2, help="동시에 실행할 모델 수(HTP 0·1에 번갈아)"
)
a = ap.parse_args()
SIZE = (1280, 720)


def detect(m, frame):
    x, r, off = resize_pad(frame, *m.size)  # ARM 전처리
    return yolox(m.run(to_rgb(x)), r, off)  # NPU → ARM 후처리


# ① 한 장씩 차례로: 해독 → 전처리 → NPU → 후처리 → 그리기
print("[1] 한 장씩 차례로 처리(NPU 1개, 기본 성능 모드)")
m = Model("yolox")
sw = Stopwatch()
video = VideoSource(a.video, SIZE, prefetch=0)
n = 0
c0, t0 = cpu_seconds(), time.perf_counter()
while True:
    with sw("해독"):
        frame = video.read()
    if frame is None:
        break
    with sw("전처리"):
        x, r, off = resize_pad(frame, *m.size)
        x = to_rgb(x)
    with sw("NPU 추론"):
        out = m.run(x)
    with sw("후처리"):
        dets = yolox(out, r, off)
    with sw("그리기"):
        draw_boxes(frame.copy(), dets)
    n += 1
wall = time.perf_counter() - t0
video.close()
sw.report(skip=5)
print(
    f"  실제 {n}장 {wall:.1f}초 → {n / wall:.1f} FPS,"
    f" CPU {(cpu_seconds() - c0) / wall:.2f}코어\n"
)

# ② 겹쳐 처리: 해독은 별도 스레드가 미리 하고, 모델 여러 개가 HTP 2개에서 동시에 추론한다
print(f"[2] 겹쳐 처리(모델 {a.workers}개, HTP 2개, 최고 성능 모드)")
pool = Pool("yolox", a.workers)
video = VideoSource(a.video, SIZE)
out_video = VideoWriter(a.save, SIZE, video.fps) if a.save else None
jobs = deque()
n, total = 0, 0


def finish(job):
    global total
    frame, fut = job
    dets = fut.result()  # 결과는 들어온 순서대로 꺼낸다
    total += len(dets)
    if out_video:
        out_video.write(draw_boxes(frame, dets))


c0, t0 = cpu_seconds(), time.perf_counter()
for frame in video:
    jobs.append((frame, pool.submit(detect, frame)))
    n += 1
    if len(jobs) >= a.workers * 2:
        finish(jobs.popleft())
while jobs:
    finish(jobs.popleft())
wall = time.perf_counter() - t0
video.close()
pool.close()
if out_video:
    out_video.close()
print(
    f"  실제 {n}장 {wall:.1f}초 → {n / wall:.1f} FPS,"
    f" CPU {(cpu_seconds() - c0) / wall:.2f}코어, 검출 {total / n:.1f}개/장"
)
if a.save:
    print(f"  결과 영상: {a.save}")
