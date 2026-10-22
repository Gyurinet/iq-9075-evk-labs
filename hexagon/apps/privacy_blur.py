# 영상 개인정보 보호: 얼굴을 찾아(NPU) 흐리게 가린다(ARM). 외부에 영상을 넘기기 전 처리에 쓴다
#   python3 hexagon/apps/privacy_blur.py samples/videos/pedestrians.mp4 --save blurred.mp4
#   python3 hexagon/apps/privacy_blur.py <사진> -o blurred.jpg       (사진 한 장)
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 저장소 맨 위(labkit)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hexagon/(qnpu)
from labkit import VideoSource, VideoWriter
from qnpu import Model, Pool, blur, people, resize_pad, to_rgb

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--save", default="blurred.mp4", help="결과 영상(mp4)")
ap.add_argument("-o", default="blurred.jpg", help="결과 사진(jpg)")
ap.add_argument("--conf", type=float, default=0.5, help="얼굴로 볼 점수(낮추면 더 많이 가린다)")
ap.add_argument("--show", action="store_true", help="가린 자리에 테두리도 그린다(확인용)")
ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()


def faces(m, img):
    x, r, off = resize_pad(img, *m.size)
    return people(m.run(to_rgb(x)), r, off, face_conf=a.conf)[0]


if a.src.lower().endswith((".jpg", ".jpeg", ".png")):
    img = cv2.imread(a.src)
    found = faces(Model("people"), img)
    cv2.imwrite(a.o, blur(img, found, a.show))
    print(f"얼굴 {len(found)}개를 가렸습니다 → {a.o}")
    sys.exit()

W, H = 1280, 720
pool = Pool("people", a.workers)
video = VideoSource(a.src, (W, H))
out = VideoWriter(a.save, (W, H), video.fps)
jobs = deque()
n, total = 0, 0
t0 = time.perf_counter()


def finish(job):
    global total
    frame, fut = job
    found = fut.result()
    total += len(found)
    out.write(blur(frame, found, a.show))


for frame in video:
    jobs.append((frame, pool.submit(faces, frame)))
    n += 1
    if len(jobs) >= a.workers * 2:
        finish(jobs.popleft())
while jobs:
    finish(jobs.popleft())
wall = time.perf_counter() - t0
video.close()
pool.close()
out.close()
print(
    f"{n}프레임을 {wall:.1f}초에 처리: {n / wall:.1f} FPS,"
    f" 프레임당 얼굴 {total / n:.1f}개 → {a.save}"
)
