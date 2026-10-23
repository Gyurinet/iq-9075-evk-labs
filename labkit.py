# 실습 공용 도구: 모델 위치, 전처리, YOLO 후처리, 결과 그리기, 시간 재기
# labs/·apps/의 스크립트가 함께 쓴다.
import atexit
import fcntl
import os
import queue
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

# OpenCV(FFmpeg)로 RTSP 카메라를 받을 때: TCP로 받고(패킷 손실이 적다), 해독 경고는 줄인다.
# 스트림 중간부터 받으면 처음 몇 장에서 참조 프레임이 없다는 경고가 나오는데 해롭지 않다
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
MODELS = Path(os.environ.get("NPU_MODELS", Path.home() / "npu-models"))

# fmt: off
COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]
# fmt: on


def model(name: str) -> str:
    """모델 파일 이름 → 전체 경로(~/npu-models, 환경 변수 NPU_MODELS로 바꿀 수 있다)"""
    p = MODELS / name
    if not p.exists():
        raise SystemExit(f"모델이 없습니다: {p}\n먼저 tools/get-models.sh 를 실행하세요.")
    return str(p)


def imagenet_labels() -> list[str]:
    return (ROOT / "data" / "imagenet_classes.txt").read_text().splitlines()


def letterbox(img, size=640, pad=114):
    """비율을 지키며 size×size에 맞추고 남는 곳을 회색으로 채운다.
    돌려주는 값: (결과, 배율, (왼쪽, 위) 여백)"""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), pad, np.uint8)
    out[top : top + nh, left : left + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_LINEAR
    )
    return out, r, (left, top)


def to_npu(img_bgr):
    """OpenCV 이미지(BGR) → NPU 입력(RGB, uint8, NHWC). 정규화는 모델 안에서 한다"""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def unletter(boxes, r, off):
    """레터박스 좌표 → 원본 이미지 좌표(x1, y1, x2, y2)"""
    b = boxes.copy()
    b[:, [0, 2]] -= off[0]
    b[:, [1, 3]] -= off[1]
    return b / r


def yolo11_decode(pred, r, off, conf=0.25, iou=0.45, classes=None):
    """YOLO11 출력 [84, 8400] → [[x1, y1, x2, y2, 점수, 분류], ...].
    NMS(겹친 상자 정리)를 ARM에서 한다"""
    p = pred.reshape(pred.shape[-2], -1).T  # [8400, 84]
    cls = p[:, 4:].argmax(1)
    score = p[np.arange(len(p)), 4 + cls]
    keep = score >= conf
    if classes is not None:
        keep &= np.isin(cls, classes)
    p, cls, score = p[keep], cls[keep], score[keep]
    if not len(p):
        return np.zeros((0, 6), np.float32)
    xywh = p[:, :4].copy()
    xywh[:, :2] -= xywh[:, 2:] / 2  # 가운데 → 왼쪽 위
    idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), score.tolist(), cls.tolist(), conf, iou)
    idx = np.array(idx, int).reshape(-1)
    b = xywh[idx]
    b[:, 2:] += b[:, :2]
    return np.column_stack([unletter(b, r, off), score[idx], cls[idx]]).astype(np.float32)


def yolo26_decode(pred, r, off, conf=0.25, classes=None):
    """YOLO26 출력 [300, 6](x1, y1, x2, y2, 점수, 분류) → 같은 형식.
    모델이 상자를 이미 정리해 NMS가 없다"""
    p = pred.reshape(-1, 6)
    keep = p[:, 4] >= conf
    if classes is not None:
        keep &= np.isin(p[:, 5].astype(int), classes)
    p = p[keep]
    return np.column_stack([unletter(p[:, :4], r, off), p[:, 4:6]]).astype(np.float32)


# fmt: off
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
    (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
    (255, 56, 132), (133, 0, 82), (255, 56, 203)
]
# fmt: on


def color(i: int):
    return PALETTE[int(i) % len(PALETTE)]


def draw_boxes(img, dets, names=COCO, thick=None):
    """검출 결과를 그린다(분류 이름과 점수)"""
    t = thick or max(2, round(min(img.shape[:2]) / 360))
    for x1, y1, x2, y2, s, c in dets:
        c = int(c)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color(c), t)
        label = f"{names[c]} {s:.2f}"
        fs = t / 3
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t // 2))
        y = max(p1[1], th + 4)
        cv2.rectangle(img, (p1[0], y - th - 4), (p1[0] + tw + 4, y), color(c), -1)
        cv2.putText(
            img,
            label,
            (p1[0] + 2, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            (255, 255, 255),
            max(1, t // 2),
            cv2.LINE_AA,
        )
    return img


class Stopwatch:
    """단계별 시간을 모은다. with sw("단계"): ... 로 잰다"""

    def __init__(self):
        self.ms: dict[str, list[float]] = {}

    @contextmanager
    def __call__(self, name):
        t = time.perf_counter()
        yield
        self.ms.setdefault(name, []).append((time.perf_counter() - t) * 1000)

    def report(self, skip=0):
        """단계별 평균(ms). 앞의 skip회(준비 시간)는 뺀다"""
        rows = []
        for k, v in self.ms.items():
            v = v[skip:] or v
            rows.append((k, float(np.mean(v)), len(v)))
        total = sum(m for _, m, _ in rows)
        for k, m, _ in rows:
            print(f"  {k:<10} {m:8.2f} ms  ({m / total * 100:4.1f}%)")
        print(f"  {'합계':<10} {total:8.2f} ms  → 한 장씩 처리하면 {1000 / total:.1f} FPS")
        return dict((k, m) for k, m, _ in rows)


def cpu_seconds():
    """이 프로그램(모든 스레드)과 하위 프로세스(GStreamer)가 지금까지 쓴 CPU 시간.
    단위는 초이며 모든 코어를 더한 값이다. 다른 프로그램이 함께 돌아도 영향을 받지 않는다"""
    t = os.times()
    total = t.user + t.system + t.children_user + t.children_system
    tick = os.sysconf("SC_CLK_TCK")
    for task in Path("/proc/self/task").iterdir():  # 아직 실행 중인 하위 프로세스
        try:
            kids = (task / "children").read_text().split()
        except OSError:
            continue
        for pid in kids:
            try:
                f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                total += (int(f[11]) + int(f[12])) / tick  # utime, stime
            except (OSError, IndexError):
                pass
    return total


# ───── 영상 입출력: 기본은 ARM 소프트웨어(OpenCV), --hw면 SoC의 하드웨어 코덱(V4L2) ─────
def _gst_source(src: str) -> list[str]:
    """입력 주소 → GStreamer 앞부분(파일 mp4, RTSP 카메라, USB 카메라)"""
    if src.startswith("rtsp://"):
        return [
            "rtspsrc",
            f"location={src}",
            "latency=200",
            "!",
            "rtph264depay",
            "!",
            "h264parse",
            "!",
            *_HWDEC,
        ]
    if src.startswith("/dev/video"):
        return ["v4l2src", f"device={src}"]
    return ["filesrc", f"location={src}", "!", "qtdemux", "!", "h264parse", "!", *_HWDEC]


# 하드웨어 디코더 출력은 기본이 퀄컴 압축 형식(NV12_Q08C)이라 일반 NV12로 받는다
_HWDEC = ["v4l2h264dec", "!", "video/x-raw,format=NV12"]


def _fps(src: str) -> float:
    """영상 파일의 초당 프레임 수(카메라·RTSP는 알 수 없으면 25)"""
    if src.startswith(("rtsp://", "/dev/video")):
        return 25.0
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps


class VideoSource:
    """영상 프레임을 BGR 배열로 하나씩 돌려준다.
    hw=False(기본): OpenCV 소프트웨어 디코더가 ARM CPU에서 해독한다.
    hw=True: 하드웨어 디코더(v4l2h264dec)가 해독하고, GStreamer가 크기를 줄여 I420으로 넘기면
             OpenCV가 BGR로 바꾼다. 검증 장비(커널 6.8.0-1080-qcom)에서는 하드웨어 코덱을 오래
             쓰다가 장비가 리셋된 적이 있어 기본으로 쓰지 않는다(가이드 5.5절)."""

    def __init__(self, src: str, size=(1280, 720), hw=False, prefetch=4):
        self.w, self.h = size
        self.hw = hw and shutil.which("gst-launch-1.0") is not None
        if self.hw:
            # 디코더 출력은 높이가 16의 배수로 늘어나 있다(1080 → 1088).
            # videoconvert가 실제 크기로 다시 담는다
            caps = f"video/x-raw,width={self.w},height={self.h}"
            cmd = ["gst-launch-1.0", "-q", *_gst_source(src), "!", "videoscale", "!", caps]
            cmd += ["!", "videoconvert", "!", "video/x-raw,format=I420", "!", "fdsink", "fd=1"]
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
            try:  # 파이프를 1 MB로 늘려 GStreamer가 덜 기다리게 한다(F_SETPIPE_SZ)
                fcntl.fcntl(self.proc.stdout.fileno(), 1031, 1 << 20)
            except OSError:
                pass
            atexit.register(self.proc.kill)  # 프로그램이 끝나면 GStreamer도 끝낸다
        else:
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                raise SystemExit(f"영상을 열 수 없습니다: {src}")
        self.fps = _fps(src)
        # prefetch > 0: 뒤에서 미리 읽어 둔다(해독·변환이 추론과 겹친다)
        self.queue, self.filler, self.stopping = None, None, False
        if prefetch:
            self.queue = queue.Queue(prefetch)
            self.filler = threading.Thread(target=self._fill, daemon=True)
            self.filler.start()

    def _fill(self):
        while not self.stopping:
            f = self._read()
            while not self.stopping:  # 가득 차 있으면 자리가 날 때까지(끝낼 때는 그만)
                try:
                    self.queue.put(f, timeout=0.1)
                    break
                except queue.Full:
                    pass
            if f is None:
                return

    def read(self):
        """다음 프레임(없으면 None)"""
        return self.queue.get() if self.queue else self._read()

    def _read(self):
        if self.hw:
            # 프레임 하나(I420: 밝기 W×H + 색 W×H/2)를 미리 만든 배열에 바로 읽는다
            yuv = np.empty((self.h * 3 // 2, self.w), np.uint8)
            view, got = memoryview(yuv).cast("B"), 0
            while got < len(view):
                k = self.proc.stdout.readinto(view[got:])
                if not k:
                    return None
                got += k
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        ok, f = self.cap.read()
        if not ok:
            return None
        if f.shape[1] != self.w or f.shape[0] != self.h:
            f = cv2.resize(f, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return f

    def __iter__(self):
        while (f := self.read()) is not None:
            yield f

    def close(self):
        # 미리 읽는 스레드를 먼저 멈춘다(읽는 도중에 디코더를 닫으면 비정상 종료한다)
        self.stopping = True
        if self.hw:
            self.proc.kill()
            self.proc.wait()
        if self.filler:
            self.filler.join(timeout=5)
        if not self.hw:
            self.cap.release()


class VideoWriter:
    """BGR 프레임을 mp4로 저장한다.
    hw=False(기본): OpenCV(FFmpeg)의 MPEG-4 인코더(ARM CPU).
    hw=True: 하드웨어 인코더 v4l2h264enc로 H.264(GStreamer)"""

    def __init__(self, path: str, size=(1280, 720), fps=24, hw=False):
        w, h = size
        self.hw = hw
        if not hw:
            self.out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            return
        raw = f"rawvideoparse width={w} height={h} format=bgr framerate={round(fps)}/1"
        enc = "videoconvert ! video/x-raw,format=NV12 ! v4l2h264enc ! h264parse ! mp4mux"
        cmd = ["gst-launch-1.0", "-q", "-e", "fdsrc", "fd=0", "!", *raw.split(), "!"]
        cmd += [*enc.split(), "!", "filesink", f"location={path}"]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        if not self.hw:
            self.out.write(frame)
            return
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        if not self.hw:
            self.out.release()
            return
        self.proc.stdin.close()
        self.proc.wait()


# ───── 객체 추적: 프레임마다 검출한 상자를 같은 물체끼리 이어 번호(ID)를 붙인다 ─────
def iou(a, b):
    """상자 a[N,4]와 b[M,4]의 겹침 비율(IoU) [N, M]"""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


class Track:
    def __init__(self, tid, det):
        self.id, self.box, self.score, self.cls = (
            tid,
            det[:4].copy(),
            float(det[4]),
            int(det[5]),
        )
        self.hits, self.missed = 1, 0
        self.path = [self.foot()]

    def foot(self):
        """발 위치(상자 아래 가운데): 바닥에 그린 선·구역과 비교할 점"""
        return float((self.box[0] + self.box[2]) / 2), float(self.box[3])


class Tracker:
    """간단한 추적기. ① 마지막 자리와 많이 겹치는 상자,
    ② 남은 것 중 가까운 상자 순서로 같은 물체를 잇는다.
    max_missed 프레임까지 안 보여도(가려짐) 번호를 지킨다"""

    def __init__(self, iou_min=0.2, max_missed=25, min_hits=3, near=1.0):
        self.iou_min, self.max_missed, self.min_hits, self.near = (
            iou_min,
            max_missed,
            min_hits,
            near,
        )
        self.tracks: list[Track] = []
        self.next_id = 1

    def _pairs(self, last, dets):
        same = np.array([t.cls for t in self.tracks])[:, None] == dets[None, :, 5].astype(int)
        m = np.where(same, iou(last, dets[:, :4]), 0)
        pairs = []
        for _ in range(min(m.shape)):  # ① 큰 겹침부터 짝짓기
            i, j = np.unravel_index(np.argmax(m), m.shape)
            if m[i, j] < self.iou_min:
                break
            pairs.append((i, j))
            m[i, :] = 0
            m[:, j] = 0
        ti = [i for i in range(len(last)) if i not in {p[0] for p in pairs}]
        dj = [j for j in range(len(dets)) if j not in {p[1] for p in pairs}]
        if ti and dj:  # ② 남은 것: 중심 거리 / 상자 크기
            c1 = (last[ti, :2] + last[ti, 2:]) / 2
            c2 = (dets[dj, :2] + dets[dj, 2:4]) / 2
            size = np.sqrt((last[ti, 2] - last[ti, 0]) * (last[ti, 3] - last[ti, 1])) + 1
            d = np.linalg.norm(c1[:, None] - c2[None], axis=2) / size[:, None]
            d[~same[np.ix_(ti, dj)]] = np.inf
            for _ in range(min(d.shape)):
                a, b = np.unravel_index(np.argmin(d), d.shape)
                if d[a, b] > self.near:
                    break
                pairs.append((ti[a], dj[b]))
                d[a, :] = np.inf
                d[:, b] = np.inf
        return pairs

    def update(self, dets):
        """dets [[x1,y1,x2,y2,점수,분류], ...] → 지금 보이는(확정된) Track 목록"""
        dets = np.asarray(dets, np.float32).reshape(-1, 6)
        used = set()
        if self.tracks and len(dets):
            for i, j in self._pairs(np.array([t.box for t in self.tracks]), dets):
                t = self.tracks[i]
                t.box, t.score, t.hits, t.missed = (
                    dets[j, :4].copy(),
                    float(dets[j, 4]),
                    t.hits + 1,
                    -1,
                )
                t.path.append(t.foot())
                used.add(j)
        for t in self.tracks:
            t.missed += 1
        for j in range(len(dets)):
            if j not in used:
                self.tracks.append(Track(self.next_id, dets[j]))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return [t for t in self.tracks if t.missed == 0 and t.hits >= self.min_hits]


def side(p, a, b):
    """점 p가 선분 a→b의 어느 쪽인지(+: 왼쪽, -: 오른쪽)"""
    return np.sign((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))


def put_text(img, text, org, scale=0.8, fg=(255, 255, 255), bg=(0, 0, 0)):
    """배경 상자를 깐 글자(영문·숫자)"""
    t = max(1, round(scale * 2))
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
    x, y = org
    cv2.rectangle(img, (x - 4, y - h - 8), (x + w + 4, y + 6), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, t, cv2.LINE_AA)
