# 실습 9: 양자화 들여다보기. 같은 ResNet50의 실수 모델과 8비트 모델을 사진마다 비교한다
#   python3 hexagon/labs/h09_quant.py samples/photos/*.jpg
#   기준: 실수 모델을 ARM CPU(32비트 실수)로 계산한 결과
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qnpu import Model, labels, model_path, to_rgb

names = labels("resnet50")
ref = Model("resnet50-float", device="cpu", threads=8)  # 기준
cases = {
    "실수 모델, NPU fp16": Model("resnet50-float"),
    "8비트 모델, ARM CPU": Model("resnet50", device="cpu", threads=8),
    "8비트 모델, NPU": Model("resnet50"),
}

# ① 모델 안을 본다: 8비트 모델의 텐서 몇 개와 배율·영점
q = Model("resnet50", device="cpu")
tensors = q.it.get_tensor_details()
int8 = [t for t in tensors if t["dtype"] in (np.int8, np.uint8)]
print(f"8비트 모델: 텐서 {len(tensors)}개 중 정수 {len(int8)}개")
weights = [t for t in int8 if len(t["quantization_parameters"]["scales"]) > 1]
acts = [t for t in int8 if len(t["shape"]) == 4 and t["shape"][0] == 1]
w = next(t for t in weights if len(t["shape"]) == 4)  # 첫 합성곱의 가중치
sc = w["quantization_parameters"]["scales"]
shape = [int(v) for v in w["shape"]]
print(f"  합성곱 가중치 {shape}: 출력 채널마다 배율 {len(sc)}개({sc.min():.4f}~{sc.max():.4f})")
for t in acts[:2]:
    s, z = t["quantization"]
    shape = [int(v) for v in t["shape"]]
    print(f"  중간값 {shape}: 배율 {s:.5f}, 영점 {z}")
size = {k: model_path(k).stat().st_size / 2**20 for k in ("resnet50-float", "resnet50")}
print(f"파일 크기: 실수 {size['resnet50-float']:.0f} MB, 8비트 {size['resnet50']:.0f} MB\n")


def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


# ② 사진마다 기준과 비교: 1순위가 같은가, 확률이 얼마나 다른가
stats = {k: {"same": 0, "diff": [], "cos": []} for k in cases}
for path in sys.argv[1:]:
    img = cv2.imread(path)
    x = to_rgb(cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA))
    base = ref.run(x.astype(np.float32) / 255)["class_logits"]
    p0 = softmax(base)
    for k, m in cases.items():
        xi = x.astype(np.float32) / 255 if m.inputs[0]["dtype"] == np.float32 else x
        out = m.run(xi)["class_logits"]
        p = softmax(out)
        st = stats[k]
        st["same"] += int(np.argmax(out) == np.argmax(base))
        st["diff"].append(abs(p[np.argmax(base)] - p0[np.argmax(base)]))
        st["cos"].append(np.dot(out, base) / (np.linalg.norm(out) * np.linalg.norm(base)))

n = len(sys.argv) - 1
print(f"사진 {n}장, 기준: 실수 모델 + ARM CPU(32비트)")
print(f"{'경우':<22}{'1순위 일치':>10}{'확률 차이(평균)':>16}{'출력 닮음(코사인)':>18}")
for k, st in stats.items():
    pad = 22 - sum(2 if ord(c) > 0x1100 else 1 for c in k) + len(k)
    print(
        f"{k:<{pad}}{st['same']:>7}/{n}{np.mean(st['diff']) * 100:>14.2f}%p"
        f"{np.mean(st['cos']):>18.4f}"
    )
