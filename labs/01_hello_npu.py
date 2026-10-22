# 실습 1: NPU에 모델을 올리고 한 번 실행해 본다
#   python3 labs/01_hello_npu.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import model
from dx_engine import InferenceEngine

ie = InferenceEngine(model("resnet50_224x224.dxnn"))       # ① 모델을 NPU에 올린다

for t in ie.get_input_tensors_info():                      # ② 입력 형식 확인
    print("입력 :", t["name"], t["shape"], t["dtype"].__name__)
for t in ie.get_output_tensors_info():
    print("출력 :", t["name"], t["shape"], t["dtype"].__name__)

x = np.zeros(ie.get_input_tensors_info()[0]["shape"][1:], np.uint8)   # ③ 검은 이미지 1장(224×224×3)
for i in range(5):
    y = ie.run([x])                                        # ④ 추론(끝날 때까지 기다린다)
    print(f"{i + 1}회: NPU 계산 {ie.get_npu_inference_time() / 1000:.2f} ms, "
          f"전체 {ie.get_latency() / 1000:.2f} ms, 출력 {y[0].shape}")
