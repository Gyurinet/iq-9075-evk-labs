# 실습 10: 영상 해독을 ARM 소프트웨어(OpenCV)와 하드웨어 디코더(V4L2)로 비교한다
#   python3 labs/10_video_decode.py samples/videos/crossing.mp4
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import VideoSource, cpu_seconds

src = sys.argv[1]


def run(label, hw, size):
    """영상 끝까지 프레임을 받기만 한다: 초당 프레임 수와 쓴 CPU 코어 수"""
    c0, t0 = cpu_seconds(), time.perf_counter()
    video = VideoSource(src, size, hw=hw, prefetch=0)
    n = sum(1 for _ in video)
    video.close()
    wall = time.perf_counter() - t0
    cores = (cpu_seconds() - c0) / wall
    print(f"{label:<34}{n:>6}{n / wall:>9.1f}{cores:>9.2f}")


print(f"{'방법':<34}{'프레임':>6}{'FPS':>9}{'CPU 코어':>9}")
run("ARM 소프트웨어(OpenCV), 1920×1080", False, (1920, 1080))
run("하드웨어 디코더, 1920×1080", True, (1920, 1080))
run("하드웨어 디코더, 1280×720", True, (1280, 720))
run("하드웨어 디코더, 640×360", True, (640, 360))
