# 여러 카메라 동시 처리: 카메라(또는 영상 파일) N대의 프레임을 실시간 속도로 받아
# NPU 하나로 검출한다
#   python3 apps/multi_stream.py samples/videos/crossing.mp4 --streams 8 --seconds 20
#   python3 apps/multi_stream.py rtsp://카메라1/stream rtsp://카메라2/stream ...
#   영상 파일은 끝나면 처음부터 다시 재생해 카메라처럼 쓴다(--streams로 같은 파일을 여러 대로)
#   처리가 밀리면 새 프레임이 올 때 아직 처리하지 못한 프레임을 버린다(실시간 우선).
#   버린 비율을 함께 보여 준다
#   하드웨어 디코더는 --hw-max 개(기본 4)까지만 연다. 검증 장비에서 1080p 디코더 8~12개를
#   NPU 최대 부하와 함께 쓰다가
#   장비가 리셋된 적이 있다. 그보다 많은 카메라는 ARM 소프트웨어 디코더(OpenCV)로 받는다
import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import VideoSource, cpu_seconds, letterbox, model, to_npu, yolo26_decode
from dx_engine import InferenceEngine

ap = argparse.ArgumentParser()
ap.add_argument("sources", nargs="+")
ap.add_argument(
    "--streams", type=int, default=0, help="영상 파일 하나를 이 수만큼 카메라로 쓴다"
)
ap.add_argument("--seconds", type=float, default=20)
ap.add_argument("--workers", type=int, default=6, help="NPU에 동시에 보내는 작업 스레드 수")
ap.add_argument(
    "--size", default="640x360", help="분석용 해상도(작을수록 ARM 변환 부담이 준다)"
)
ap.add_argument("--hw-max", type=int, default=4, help="하드웨어 디코더로 받을 카메라 수 상한")
a = ap.parse_args()
sources = a.sources * a.streams if a.streams else a.sources
W, H = (int(v) for v in a.size.split("x"))
ie = InferenceEngine(model("yolo26-n_640x640.dxnn"))
stop = threading.Event()


class Camera:
    """한 대의 카메라: 실시간 속도로 프레임을 받아 '가장 최근 프레임' 자리에 둔다"""

    def __init__(self, i, src):
        self.i, self.src = i, src
        self.hw = i < a.hw_max
        self.frame, self.got, self.done, self.dropped, self.objects = None, 0, 0, 0, 0
        self.lock = threading.Lock()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while not stop.is_set():
            video = VideoSource(self.src, (W, H), hw=self.hw, prefetch=2)
            t0, k = time.perf_counter(), 0
            for f in video:
                if stop.is_set():
                    break
                k += 1
                # 실제 카메라 속도로
                time.sleep(max(0.0, t0 + k / video.fps - time.perf_counter()))
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


cams = []
for i, s in enumerate(sources):  # 디코더를 한꺼번에 열지 않고 0.5초 간격으로 연다
    cams.append(Camera(i, s))
    time.sleep(0.5)
turn = 0
turn_lock = threading.Lock()


def worker():
    """카메라를 돌아가며 새 프레임을 가져와 검출한다"""
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
        x, r, off = letterbox(frame, 640)
        dets = yolo26_decode(ie.run([to_npu(x)])[0], r, off, 0.35)
        with cam.lock:
            cam.done += 1
            cam.objects += len(dets)


threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
for t in threads:
    t.start()
time.sleep(3)  # 준비(디코더 시작) 시간은 빼고 잰다
for c in cams:
    with c.lock:
        c.got = c.done = c.dropped = c.objects = 0
c0, t0 = cpu_seconds(), time.perf_counter()
time.sleep(a.seconds)
wall = time.perf_counter() - t0
cores = (cpu_seconds() - c0) / wall
stop.set()
for t in threads:  # 하던 추론을 마치고 끝낸다(NPU를 쓰는 중에 끝내면 비정상 종료)
    t.join()

got = sum(c.got for c in cams)
done = sum(c.done for c in cams)
print(f"카메라 {len(cams)}대, {a.size}, {wall:.0f}초")
for c in cams:
    print(
        f"  #{c.i + 1}({'HW' if c.hw else 'SW'}): 받은 {c.got / wall:5.1f} fps, "
        f"처리 {c.done / wall:5.1f} fps, 버림 {c.dropped / max(1, c.got) * 100:4.1f}%, "
        f"물체 {c.objects / max(1, c.done):.1f}개/장"
    )
print(
    f"합계: 처리 {done / wall:.1f} fps / 받은 {got / wall:.1f} fps "
    f"({min(100, done / max(1, got) * 100):.0f}%), CPU {cores:.1f}코어"
)
