# 실습 1: NPU에 모델을 올리고 한 번 실행해 본다
#   python3 labs/01_hello_npu.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import model
from dx_engine import InferenceEngine

# ① 모델을 NPU에 올린다
ie = InferenceEngine(model("resnet50_224x224.dxnn"))

# ② 입력·출력 형식을 확인한다
for t in ie.get_input_tensors_info():
    print("입력 :", t["name"], t["shape"], t["dtype"].__name__)
for t in ie.get_output_tensors_info():
    print("출력 :", t["name"], t["shape"], t["dtype"].__name__)

# ③ 입력 하나를 만든다: 검은 이미지 1장(224×224×3, uint8)
shape = ie.get_input_tensors_info()[0]["shape"]  # [1, 224, 224, 3]
x = np.zeros(shape[1:], np.uint8)

# ④ 추론한다(끝날 때까지 기다린다). 시간은 마이크로초 단위로 돌려준다
for i in range(5):
    y = ie.run([x])
    npu_ms = ie.get_npu_inference_time() / 1000
    total_ms = ie.get_latency() / 1000
    print(f"{i + 1}회: NPU 계산 {npu_ms:.2f} ms, 전체 {total_ms:.2f} ms, 출력 {y[0].shape}")
