# 추론 서버 호출 예제와 부하 시험(표준 라이브러리만 쓰므로 어느 PC에서나 실행된다)
#   python3 apps/infer_client.py http://<장비 IP>:8080/detect samples/images/crossing.jpg
#   python3 apps/infer_client.py http://<장비 IP>:8080/detect samples/images/crossing.jpg --bench 200 --clients 8
import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("image")
ap.add_argument("--bench", type=int, default=0, help="이 횟수만큼 보내 처리량을 잰다")
ap.add_argument("--clients", type=int, default=4, help="동시에 보내는 수")
a = ap.parse_args()
body = open(a.image, "rb").read()


def call():
    t = time.perf_counter()
    req = urllib.request.Request(a.url, data=body, headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.load(r)
    return res, (time.perf_counter() - t) * 1000


if not a.bench:
    res, ms = call()
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"왕복 {ms:.1f} ms (서버 처리 {res['ms']} ms)")
else:
    call()                                                 # 준비
    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.clients) as pool:
        lat = [ms for _, ms in pool.map(lambda _: call(), range(a.bench))]
    wall = time.perf_counter() - t0
    lat.sort()
    print(f"{a.bench}회, 동시 {a.clients}개: {a.bench / wall:.1f} 요청/초, "
          f"응답 시간 중앙값 {statistics.median(lat):.1f} ms, 95% {lat[int(len(lat) * 0.95) - 1]:.1f} ms")
