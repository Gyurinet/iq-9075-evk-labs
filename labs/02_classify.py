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
_, h, w, _ = ie.get_input_tensors_info()[0]["shape"]      # [1, 224, 224, 3]

for path in sys.argv[1:]:
    img = cv2.imread(path)
    x = to_npu(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))   # ARM: 크기 맞추기, BGR→RGB
    logits = ie.run([x])[0].reshape(-1)                                 # NPU: 추론
    p = np.exp(logits - logits.max()); p /= p.sum()                     # ARM: 점수 → 확률(softmax)
    top = np.argsort(p)[::-1][:3]
    print(Path(path).name, " | ".join(f"{labels[i]} {p[i] * 100:.1f}%" for i in top),
          f"(NPU {ie.get_npu_inference_time() / 1000:.2f} ms)")
