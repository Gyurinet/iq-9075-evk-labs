# 실습 2: 이미지 분류(ResNet50, ImageNet 1000종)
#   python3 labs/02_classify.py samples/images/forklift.jpg [이미지 …]
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import imagenet_labels, model, to_npu
from dx_engine import InferenceEngine

labels = imagenet_labels()
ie = InferenceEngine(model("resnet50_224x224.dxnn"))
_, h, w, _ = ie.get_input_tensors_info()[0]["shape"]  # [1, 224, 224, 3]

for path in sys.argv[1:]:
    img = cv2.imread(path)
    # ARM: 크기를 맞추고 BGR(OpenCV 순서)을 RGB로 바꾼다
    x = to_npu(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
    # NPU: 추론. 출력은 1000종 각각의 점수(logit)
    logits = ie.run([x])[0].reshape(-1)
    # ARM: 점수를 확률로 바꾸고(softmax) 큰 순서로 3개
    p = np.exp(logits - logits.max())
    p /= p.sum()
    top = np.argsort(p)[::-1][:3]
    result = " | ".join(f"{labels[i]} {p[i] * 100:.1f}%" for i in top)
    npu_ms = ie.get_npu_inference_time() / 1000
    print(f"{Path(path).name} {result} (NPU {npu_ms:.2f} ms)")
