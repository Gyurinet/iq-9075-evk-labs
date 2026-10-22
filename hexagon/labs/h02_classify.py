# 실습 2: 이미지 분류(ResNet50, ImageNet 1000종)를 내장 NPU에서
#   python3 hexagon/labs/h02_classify.py samples/images/forklift.jpg [이미지 …]
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qnpu import Model, labels, to_rgb

names = labels("resnet50")
m = Model("resnet50")
h, w = m.size

for path in sys.argv[1:]:
    img = cv2.imread(path)
    # ARM: 크기를 맞추고 RGB로. 0~1 정규화와 평균·표준편차 처리는 모델 안에 있다
    x = to_rgb(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
    logits = m.run(x)["class_logits"]  # NPU
    p = np.exp(logits - logits.max())  # ARM: 확률로
    p /= p.sum()
    top = np.argsort(p)[::-1][:3]
    result = " | ".join(f"{names[i]} {p[i] * 100:.1f}%" for i in top)
    print(f"{Path(path).name} {result} (NPU {m.last_ms:.2f} ms)")
