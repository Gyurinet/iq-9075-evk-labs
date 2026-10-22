# 모델 들여다보기: 입력·출력, 연산 종류별 개수, 내장 NPU(QNN 델리게이트)에 올라가는지
#   python3 hexagon/ops.py ocr-rec          (qnpu.py의 이름 또는 ~/qai-models 아래 경로)
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from qnpu import Model, model_path

name = sys.argv[1] if len(sys.argv) > 1 else "resnet50"
# CPU 가속(XNNPACK)을 끄고 읽어야 모델에 들어 있는 연산 그대로 보인다
raw = Interpreter(
    model_path=str(model_path(name)),
    experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
)
for kind, details in (("입력", raw.get_input_details()), ("출력", raw.get_output_details())):
    for d in details:
        shape = [int(v) for v in d["shape"]]
        print(f"{kind} {d['name']}: {shape} {d['dtype'].__name__}")
ops = Counter(o["op_name"] for o in raw._get_ops_details())
print(f"연산 {sum(ops.values())}개:", ", ".join(f"{k} {v}" for k, v in ops.most_common()))
try:
    Model(name)
    print("NPU: 올라감")
except RuntimeError:
    print("NPU: 올라가지 않음(QNN_VERBOSE=1로 실행하면 이유가 보인다)")
