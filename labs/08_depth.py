# 실습 8: 한 장의 사진으로 거리(깊이) 추정(Depth Anything V2 Small)
#   python3 labs/08_depth.py <이미지> [-o 결과.jpg]
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labkit import model
from dx_engine import InferenceEngine

path = sys.argv[1]
out = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else "depth.jpg"
ie = InferenceEngine(model("depthanythingv2-vits_224x224.dxnn"))

img = cv2.imread(path)
# 이 모델은 정규화를 ARM에서 한다: float32, NCHW, ImageNet 평균·표준편차
x = cv2.cvtColor(cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
x = np.ascontiguousarray(((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None], np.float32)
d = ie.run([x])[0][0]                                     # [224, 224], 가까울수록 큰 값

lo, hi = np.percentile(d, [2, 98])                        # 튀는 값은 잘라 내고 0~1로
d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
d = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
vis = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
cv2.imwrite(out, np.hstack([img, vis]))
print(f"NPU {ie.get_npu_inference_time() / 1000:.2f} ms, 저장: {out}")
