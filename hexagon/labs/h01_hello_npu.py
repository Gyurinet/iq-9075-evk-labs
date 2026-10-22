# 실습 1: 내장 NPU(Hexagon)에 모델을 올리고 한 번 실행해 본다
#   python3 hexagon/labs/h01_hello_npu.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qnpu import Model

# ① 모델을 NPU에 올린다. 처음에는 NPU용 그래프를 만드느라 시간이 걸리고, 다음부터는 캐시를 쓴다
m = Model("resnet50")
print(f"모델 올리기: {m.load_s:.2f}초")

# ② 입력·출력 형식(양자화된 8비트 모델: uint8 입력, 출력은 배율·영점으로 실수로 바꾼다)
for kind, details in (("입력", m.inputs), ("출력", m.outputs)):
    for d in details:
        shape = [int(v) for v in d["shape"]]
        scale, zero = d["quantization"]
        print(
            f"{kind} : {d['name']} {shape} {d['dtype'].__name__}, 배율 {scale:.6f}, 영점 {zero}"
        )

# ③ 검은 이미지 1장으로 5번 추론
x = np.zeros((*m.size, 3), np.uint8)
for i in range(5):
    y = m.run(x)
    print(f"{i + 1}회: {m.last_ms:.2f} ms, 출력 {y['class_logits'].shape}")
