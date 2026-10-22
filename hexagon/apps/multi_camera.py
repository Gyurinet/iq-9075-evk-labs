# 여러 카메라 동시 처리: 카메라(또는 영상 파일) N대의 프레임을 실시간 속도로 받아
# 내장 NPU로 검출한다
#   python3 hexagon/apps/multi_camera.py samples/videos/crossing.mp4 --streams 8 --seconds 20
#   python3 hexagon/apps/multi_camera.py rtsp://카메라1/stream rtsp://카메라2/stream ...
#   영상 파일은 끝나면 처음부터 다시 재생해 카메라처럼 쓴다(--streams로 같은 파일을 여러 대로)
#   처리가 밀리면 아직 처리하지 못한 프레임을 버리고 새 프레임을 쓴다(실시간 우선)
import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 저장소 맨 위(labkit)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hexagon/(qnpu)
from labkit import VideoSource, cpu_seconds
from qnpu import Model, people, resize_pad, to_rgb, yolox

ap = argparse.ArgumentParser()
ap.add_argument("sources", nargs="+")
ap.add_argument("--streams", type=int, default=0, help="영상 파일 하나를 이 수만큼 카메라로")
ap.add_argument("--seconds", type=float, default=20)
ap.add_argument("--model", choices=["yolox", "people"], default="yolox")
ap.add_argument("--workers", type=int, default=4, help="작업 스레드(모델) 수, HTP 0·1에 번갈아")
ap.add_argument("--size", default="640x360", help="분석용 해상도(작을수록 ARM 부담이 준다)")
a = ap.parse_args()
sources = a.sources * a.streams if a.streams else a.sources
W, H = (int(v) for v in a.size.split("x"))
stop = threading.Event()


class Camera:
    """한 대의 카메라: 실시간 속도로 프레임을 받아 '가장 최근 프레임' 자리에 둔다"""

    def __init__(self, i, src):
        self.i, self.src = i, src
        self.frame, self.got, self.done, self.dropped, self.objects = None, 0, 0, 0, 0
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        live = self.src.startswith(("rtsp://", "/dev/video"))
        while not stop.is_set():
            video = VideoSource(self.src, (W, H), prefetch=2)
            t0, k = time.perf_counter(), 0
            for f in video:
                if stop.is_set():
                    break
                k += 1
                if not live:  # 영상 파일은 실제 카메라 속도로 맞춰 읽는다
                    late = time.perf_counter() - (t0 + k / video.fps)
                    if (
                        late > 0.5
                    ):  # 해독이 밀리면 따라잡으려 빨리 보내지 않는다(실제 카메라처럼)
                        t0 += late
                    time.sleep(max(0.0, -late))
                with self.lock:
                    if self.frame is not None:
                        self.dropped += 1  # 처리 전에 다음 프레임이 왔다
                    self.frame = f
                    self.got += 1
            video.close()

    def take(self):
        with self.lock:
            f, self.frame = self.frame, None
            return f


def analyze(m, frame):
    x, r, off = resize_pad(frame, *m.size)
    out = m.run(to_rgb(x))
    return len(yolox(out, r, off)) if a.model == "yolox" else len(people(out, r, off)[1])


models = [Model(a.model, htp=k % 2, perf="burst") for k in range(a.workers)]
cams = []
for i, s in enumerate(sources):  # 디코더를 한꺼번에 열지 않고 0.5초 간격으로 연다
    cams.append(Camera(i, s))
    time.sleep(0.5)
turn = 0
turn_lock = threading.Lock()


def worker(m):
    """카메라를 돌아가며 새 프레임을 가져와 자기 모델로 검출한다"""
    global turn
    while not stop.is_set():
        frame = cam = None
        with turn_lock:
            for _ in range(len(cams)):
                c = cams[turn % len(cams)]
                turn += 1
                frame = c.take()
                if frame is not None:
                    cam = c
                    break
        if frame is None:
            time.sleep(0.002)
            continue
        n = analyze(m, frame)
        with cam.lock:
            cam.done += 1
            cam.objects += n


threads = [threading.Thread(target=worker, args=(m,), daemon=True) for m in models]
for t in threads:
    t.start()
time.sleep(3)  # 준비(디코더 시작) 시간은 빼고 잰다
# 스레드가 수백 개면 CPU 시간 읽기에 몇 초가 걸린다. 프레임을 세는 구간과 겹치지 않게 한다
c0, tc0 = cpu_seconds(), time.perf_counter()
for c in cams:
    with c.lock:
        c.got = c.done = c.dropped = c.objects = 0
t0 = time.perf_counter()
time.sleep(a.seconds)
wall = time.perf_counter() - t0
counts = [(c.got, c.done, c.dropped, c.objects) for c in cams]  # 이 순간의 값으로 낸다
cores = (cpu_seconds() - c0) / (time.perf_counter() - tc0)
stop.set()
for t in threads:  # 하던 추론을 마치고 끝낸다
    t.join()
for c in cams:  # 카메라도 받던 프레임을 마치고 닫는다
    c.thread.join(timeout=10)

got = sum(k[0] for k in counts)
done = sum(k[1] for k in counts)
print(f"카메라 {len(cams)}대, {a.size}, {a.model}, {wall:.0f}초")
for c, (g, d, drop, obj) in zip(cams, counts, strict=True):
    print(
        f"  #{c.i + 1}: 받은 {g / wall:5.1f} fps, 처리 {d / wall:5.1f} fps,"
        f" 버림 {drop / max(1, g) * 100:4.1f}%, 물체 {obj / max(1, d):.1f}개/장"
    )
print(
    f"합계: 처리 {done / wall:.1f} fps / 받은 {got / wall:.1f} fps"
    f" ({min(100, done / max(1, got) * 100):.0f}%), CPU {cores:.1f}코어"
)
