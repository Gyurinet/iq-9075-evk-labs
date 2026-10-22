# arm-npu-labs

ARM 프로세서와 NPU를 함께 쓰는 엣지 AI 실습 예제입니다. 「ARM + NPU 엣지 AI 실습 가이드」(gyurinet)의 예제 저장소이며,
Qualcomm IQ-9075 EVK(Cortex-A78C 8코어, Ubuntu 24.04)에 DEEPX DX-M1 NPU 모듈을 꽂은 장비에서 검증했습니다.

ARM이 영상 해독·전처리·후처리·서비스를 맡고, NPU가 신경망 계산을 맡는 구성을 실습합니다.

## 준비

DX-M1 드라이버(`dxrt-driver-dkms`)와 런타임(`libdxrt-bin`, DX-RT 3.4.2), 모듈 펌웨어 2.7.0 이상이 설치되어 있어야 합니다.
설치 순서는 가이드 2장에 있습니다.

```bash
python3 -m venv ~/npu-venv
~/npu-venv/bin/pip install /usr/share/libdxrt-bin/python/dx_engine-3.4.2-cp312-cp312-manylinux_2_27_aarch64.whl \
    numpy opencv-python-headless onnxruntime
bash tools/get-models.sh          # 모델을 ~/npu-models 로 내려받기
source ~/npu-venv/bin/activate
python3 labs/01_hello_npu.py
```

## 구성

| 폴더 | 파일 | 내용 |
|---|---|---|
| `labs/` | `01_hello_npu.py` | 모델을 NPU에 올리고 한 번 실행 |
| | `02_classify.py` | 이미지 분류(ResNet50) |
| | `03_cpu_vs_npu.py` | 같은 모델을 ARM CPU(ONNX Runtime)와 NPU에서 비교 |
| | `04_pipeline.py` | 영상 검출의 단계별 시간과 겹쳐 처리하기 |
| | `05_detect.py` ~ `08_depth.py` | 검출, 인스턴스 분할, 자세 추정, 깊이 추정 |
| | `09_npu_cores.py` | NPU 코어 나눠 쓰기 |
| `apps/` | `line_counter.py` | 가상의 선을 지나는 사람·차량 세기 |
| | `zone_guard.py` | 구역 진입·체류 감시와 기록 |
| | `privacy_blur.py` | 영상 속 얼굴 가리기 |
| | `infer_server.py`, `infer_client.py` | 사내 추론 서버(HTTP)와 호출·부하 시험 |
| | `photo_search.py` | 비슷한 사진·중복 사진 찾기(CLIP) |
| | `multi_stream.py` | 여러 카메라 동시 처리 |
| `tools/` | `get-models.sh`, `bench.sh` | 모델 내려받기, 모델별 성능 표 |
| `labkit.py` | | 공용 도구(전처리, YOLO 후처리, 영상 입출력, 추적기) |
| `samples/` | | 예제 사진·영상(퍼블릭 도메인·CC0, 출처는 `samples/SOURCES.md`) |

## 모델과 라이선스

모델(`.dxnn`, `.onnx`)은 이 저장소에 들어 있지 않습니다. `tools/get-models.sh`가 DEEPX 모델 저장소에서 내려받습니다.
DEEPX가 오픈소스 모델을 변환해 배포하는 평가·개발용 모델이므로, 제품에 넣어 배포할 때는 원 모델의 라이선스를 확인하세요.

이 저장소의 코드 라이선스는 `LICENSE`를 따릅니다.
