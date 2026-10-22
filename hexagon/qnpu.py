# IQ-9075 내장 NPU(Hexagon HTP) 공용 도구: LiteRT + QNN 델리게이트로 모델을 실행하고
# 결과를 해석한다. 영상 입출력·추적·그리기는 저장소 맨 위의 labkit.py를 쓴다.
import atexit
import ctypes
import os
import queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import cpu_seconds  # noqa: F401  (예제에서 qnpu로 함께 쓴다)

MODELS = Path(os.environ.get("QAI_MODELS", Path.home() / "qai-models"))
CACHE = Path(os.environ.get("QNN_CACHE", Path.home() / ".cache" / "qnn-delegate"))

# 모델 이름 → 파일(Qualcomm AI Hub 배포본)
FILES = {
    "resnet50": "resnet50-tflite-w8a8/resnet50.tflite",
    "resnet50-float": "resnet50-tflite-float/resnet50.tflite",
    "mobilenet_v2": "mobilenet_v2-tflite-w8a8/mobilenet_v2.tflite",
    "yolox": "yolox-tflite-w8a8/yolox.tflite",
    "people": "foot_track_net-tflite-w8a8/foot_track_net.tflite",
    "ppe": "gear_guard_net-tflite-w8a8/gear_guard_net.tflite",
    "ffnet": "ffnet_40s-tflite-w8a8/ffnet_40s.tflite",
    "depth": "depth_anything_v2-tflite-float/depth_anything_v2.tflite",
    "clip": "openai_clip-tflite-float/openai_clip.tflite",
}


_libc = ctypes.CDLL(None)


@contextmanager
def _quiet():
    """델리게이트가 모델을 준비하며 찍는 내부 기록(표준 출력·오류)을 숨긴다.
    QNN_VERBOSE=1이면 그대로 보인다"""
    if os.environ.get("QNN_VERBOSE"):
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.dup2(null, 2)
    try:
        yield
    finally:
        _libc.fflush(None)  # C 라이브러리에 남은 출력까지 버린다
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, null):
            os.close(fd)


@atexit.register
def _quiet_exit():
    """끝날 때 NPU 연결을 닫으며 찍는 기록도 숨긴다"""
    if not os.environ.get("QNN_VERBOSE"):
        sys.stdout.flush()
        _libc.fflush(None)
        os.dup2(os.open(os.devnull, os.O_WRONLY), 1)


def model_path(name: str) -> Path:
    p = MODELS / FILES.get(name, name)
    if not p.exists():
        raise SystemExit(f"모델이 없습니다: {p}\n먼저 hexagon/get-models.sh 를 실행하세요.")
    return p


def labels(name: str) -> list[str]:
    """모델 폴더의 labels.txt"""
    return (model_path(name).parent / "labels.txt").read_text().splitlines()


class Model:
    """LiteRT 모델 하나. device="npu"면 Hexagon NPU(HTP), "cpu"면 ARM CPU(XNNPACK)에서 실행한다.
    htp: 쓸 HTP 번호(0 또는 1), perf: "default" 또는 "burst"(클록을 최대로)"""

    def __init__(self, name, device="npu", htp=0, perf="default", threads=None):
        path = model_path(name)
        self.name, self.device = name, device
        t0 = time.perf_counter()
        with _quiet():
            delegates = []
            if device == "npu":
                CACHE.mkdir(parents=True, exist_ok=True)
                probe = Interpreter(model_path=str(path))
                is_float = probe.get_input_details()[0]["dtype"] == np.float32
                options = {
                    "backend_type": "htp",
                    "htp_device_id": str(htp),
                    "htp_precision": "1" if is_float else "0",  # 실수 모델은 fp16으로
                    "htp_performance_mode": "2" if perf == "burst" else "0",
                    "cache_dir": str(CACHE),  # 처음 한 번 만든 그래프를 다시 쓴다
                }
                delegates.append(load_delegate("libQnnTFLiteDelegate.so", options=options))
            self.it = Interpreter(
                model_path=str(path), experimental_delegates=delegates, num_threads=threads
            )
            self.it.allocate_tensors()
        self.load_s = time.perf_counter() - t0
        self.inputs = self.it.get_input_details()
        self.outputs = self.it.get_output_details()
        self.last_ms = 0.0

    def __del__(self):
        """모델을 내릴 때 NPU 연결을 닫으며 찍는 기록을 숨긴다"""
        try:
            with _quiet():
                self.__dict__.pop("it", None)
        except Exception:
            pass

    @property
    def size(self):
        """첫 입력의 (높이, 폭)"""
        return tuple(int(v) for v in self.inputs[0]["shape"][1:3])

    def run(self, *arrays):
        """입력 배열(배치 차원 없이) → {출력 이름: 양자화를 푼 float32 배열(배치 차원 뺌)}"""
        for d, a in zip(self.inputs, arrays, strict=True):
            self.it.set_tensor(d["index"], np.ascontiguousarray(a[None], dtype=d["dtype"]))
        t = time.perf_counter()
        self.it.invoke()
        self.last_ms = (time.perf_counter() - t) * 1000
        out = {}
        for d in self.outputs:
            x = self.it.get_tensor(d["index"])
            x = x[0] if x.shape[0] == 1 else x
            scale, zero = d["quantization"]
            out[d["name"]] = (x.astype(np.float32) - zero) * scale if scale else x
        return out


class Pool:
    """같은 모델을 여러 개 올려(HTP 0·1에 번갈아) 스레드로 동시에 실행한다.
    LiteRT의 실행(invoke)은 끝날 때까지 기다리므로, 겹쳐 처리하려면 모델이 여러 개 있어야 한다.
    submit(fn, 인자…)은 fn(모델, 인자…)을 빈 모델로 실행하고 Future를 돌려준다"""

    def __init__(self, name, workers=2, perf="burst"):
        self.free = queue.SimpleQueue()
        for k in range(workers):
            self.free.put(Model(name, htp=k % 2, perf=perf))
        self.pool = ThreadPoolExecutor(workers)

    def submit(self, fn, *args):
        def job():
            m = self.free.get()
            try:
                return fn(m, *args)
            finally:
                self.free.put(m)

        return self.pool.submit(job)

    def close(self):
        self.pool.shutdown()


# ───── 전처리 ─────
def resize_pad(img, h, w):
    """비율을 지키며 h×w에 맞추고 남는 곳을 검게 채운다(가운데). → 결과, 배율, (왼쪽, 위)"""
    r = min(h / img.shape[0], w / img.shape[1])
    nh, nw = round(img.shape[0] * r), round(img.shape[1] * r)
    out = np.zeros((h, w, 3), np.uint8)
    top, left = (h - nh) // 2, (w - nw) // 2
    out[top : top + nh, left : left + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_AREA
    )
    return out, r, (left, top)


def to_rgb(img):
    """OpenCV 이미지(BGR) → 모델 입력(RGB, uint8). 0~1 정규화는 모델 안에 있다"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def undo(boxes, r, off):
    """resize_pad 좌표 → 원본 좌표(x1, y1, x2, y2)"""
    b = np.array(boxes, np.float32).reshape(-1, 4).copy()
    b[:, [0, 2]] -= off[0]
    b[:, [1, 3]] -= off[1]
    return b / r


def nms(boxes, scores, classes, iou=0.5):
    """겹친 상자 정리(NMS). 분류가 같은 것끼리만 비교한다. 남길 번호를 돌려준다"""
    if not len(boxes):
        return np.zeros(0, int)
    xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
    idx = cv2.dnn.NMSBoxesBatched(
        xywh, list(map(float, scores)), list(map(int, classes)), 0.0, iou
    )
    return np.array(idx, int).reshape(-1)


# ───── 모델별 후처리 ─────
def yolox(out, r, off, conf=0.4, iou=0.5, classes=None):
    """YOLOX 출력(상자 8400개, 점수, 분류) → [[x1, y1, x2, y2, 점수, 분류], ...]"""
    b, s, c = out["boxes"], out["scores"], out["class_idx"].astype(int)
    keep = s >= conf
    if classes is not None:
        keep &= np.isin(c, classes)
    b, s, c = b[keep], s[keep], c[keep]
    i = nms(b, s, c, iou)
    if not len(i):
        return np.zeros((0, 6), np.float32)
    return np.column_stack([undo(b[i], r, off), s[i], c[i]]).astype(np.float32)


def people(out, r, off, face_conf=0.6, person_conf=0.7, stride=4):
    """사람·얼굴 모델(FootTrackNet) 출력 → (얼굴 [[x1,y1,x2,y2,점수]],
    사람 [[x1,y1,x2,y2,점수]], 관절 [사람 수, 17, (x, y, 보임)]). 관절 순서는 COCO이고,
    이 모델이 잘 찾는 것은 0(코)과 15·16(발목)이다. 발목은 바닥 위치(구역·선 판정)에 쓴다"""
    hm, tlrb = out["heatmap"], out["bbox"]
    lm, vis = out["landmark"], out["landmark_visibility"]
    result = []
    for cls, conf in ((0, face_conf), (1, person_conf)):
        ys, xs = np.where(hm[..., cls] > conf)
        if not len(ys):
            result.append((np.zeros((0, 5), np.float32), np.zeros((0, 17, 3), np.float32)))
            continue
        d = tlrb[ys, xs, cls * 4 : cls * 4 + 4]  # 중심에서 왼쪽·위·오른쪽·아래까지 거리
        boxes = np.stack([xs - d[:, 0], ys - d[:, 1], xs + d[:, 2], ys + d[:, 3]], 1) * stride
        scores = hm[ys, xs, cls]
        i = nms(boxes, scores, np.zeros(len(boxes)), 0.2 if cls == 0 else 0.5)
        kps = np.zeros((len(i), 17, 3), np.float32)
        if cls == 1:
            k = lm[ys[i], xs[i]]
            kps[..., 0] = ((k[:, :17] + xs[i, None]) * stride - off[0]) / r
            kps[..., 1] = ((k[:, 17:] + ys[i, None]) * stride - off[1]) / r
            kps[..., 2] = vis[ys[i], xs[i]]
        result.append((np.column_stack([undo(boxes[i], r, off), scores[i]]), kps))
    (faces, _), (persons, kps) = result
    return faces, persons, kps


def ppe(out, conf=0.45):
    """안전 장비 모델(GearGuardNet) 출력 → {"helmet": 최고 점수, "vest": 최고 점수}"""
    s, c = out["scores"], out["class_idx"].astype(int)
    return {"helmet": float(s[c == 0].max(initial=0)), "vest": float(s[c == 1].max(initial=0))}


def blur(img, faces, show=False):
    """얼굴 상자를 조금 넓혀(머리카락·귀까지) 모자이크 + 흐림. show면 테두리도 그린다"""
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in np.asarray(faces)[:, :4]:
        bw, bh = x2 - x1, y2 - y1
        x1, y1 = int(max(0, x1 - bw * 0.2)), int(max(0, y1 - bh * 0.25))
        x2, y2 = int(min(w, x2 + bw * 0.2)), int(min(h, y2 + bh * 0.15))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = img[y1:y2, x1:x2]
        small = cv2.resize(
            roi,
            (max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)),
            interpolation=cv2.INTER_AREA,
        )
        big = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        img[y1:y2, x1:x2] = cv2.GaussianBlur(big, (0, 0), 3)
        if show:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 2)
    return img
