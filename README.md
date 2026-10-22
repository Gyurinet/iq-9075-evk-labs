# arm-npu-labs

ARM 프로세서와 NPU를 함께 쓰는 엣지 AI 실습 예제입니다. gyurinet의 두 가이드가 함께 쓰는 예제 저장소이며,
Qualcomm IQ-9075 EVK(Cortex-A78C 8코어, Ubuntu 24.04)에서 검증했습니다.

| 가이드 | NPU | 예제 폴더 |
|---|---|---|
| 「ARM + 내장 NPU 엣지 AI 실습 가이드」 | IQ-9075에 들어 있는 Qualcomm Hexagon NPU(LiteRT + QNN) | `hexagon/` |
| 「ARM + NPU 엣지 AI 실습 가이드」 | PCIe로 꽂은 DEEPX DX-M1 NPU 모듈(DX-RT) | `labs/`, `apps/`, `tools/` |

ARM이 영상 해독·전처리·후처리·서비스를 맡고, NPU가 신경망 계산을 맡는 구성을 실습합니다.

## 내장 NPU(Hexagon) 예제

IQ-9075의 Qualcomm 패키지(`qairt-libs`, `qairt-dsp-binaries`, `qairt-tools`, `tensorflow-lite-qcom-apps`)가 설치되어 있어야 합니다.
설치 순서는 「ARM + 내장 NPU 엣지 AI 실습 가이드」 2장에 있습니다.

```bash
python3 -m venv ~/qai-venv
~/qai-venv/bin/pip install -r hexagon/requirements.txt
bash hexagon/get-models.sh        # Qualcomm AI Hub 모델을 ~/qai-models 로 내려받기
source ~/qai-venv/bin/activate
python3 hexagon/labs/h01_hello_npu.py
```

| 폴더 | 파일 | 내용 |
|---|---|---|
| `hexagon/labs/` | `h01_hello_npu.py`, `h02_classify.py` | 모델을 NPU에 올리기, 이미지 분류 |
| | `h03_cpu_vs_npu.py` | 같은 모델을 ARM CPU와 NPU(HTP 1개·2개)에서 비교 |
| | `h04_detect.py`, `h05_pipeline.py` | 객체 검출, 영상의 단계별 시간과 겹쳐 처리 |
| | `h06_people.py` ~ `h08_depth.py` | 사람·얼굴, 도로 장면 분할, 깊이 추정 |
| | `h09_quant.py` | 양자화 들여다보기(실수 모델과 8비트 모델) |
| | `h10_pose.py` | 두 단계 모델로 자세 추정(사람 찾기 → 관절) |
| | `h11_two_models.py` | 서로 다른 모델 두 개를 HTP 2개에 나눠 동시에 |
| | `h12_super_res.py` | 화질 높이기(초해상도) |
| `hexagon/apps/` | `ppe_check.py` | 안전모·안전 조끼 착용 점검 |
| | `people_counter.py` | 가상의 선을 지나는 사람 세기 |
| | `zone_guard.py` | 구역 진입·체류 감시와 기록 |
| | `privacy_blur.py` | 영상 속 얼굴 가리기 |
| | `cutout.py` | 사람 오려내기(배경 흐리기·바꾸기, 사람 전체 가리기) |
| | `photo_search.py` | 영어 문장·사진으로 사진 찾기(CLIP) |
| | `sort_photos.py` | 학습 없이 사진 나누기(CLIP) |
| | `read_text.py` | 글자 읽기(OCR: 검출은 NPU, 인식은 ARM CPU) |
| | `sound_watch.py` | 소리 감시(사이렌·경보·경적 등 AudioSet 521종) |
| | `infer_server.py` | 사내 추론 서버(HTTP, `apps/infer_client.py`로 호출) |
| | `multi_camera.py` | 여러 카메라 동시 처리 |
| `hexagon/` | `qnpu.py`, `clip_tokenizer.py` | 공용 도구(모델 실행, 후처리), CLIP 글 토크나이저 |
| | `get-models.sh`, `bench.py`, `soak.py`, `temps.sh` | 모델 내려받기, 성능 표, 오래 돌리기 시험, 온도 |

## DX-M1 예제

### 준비

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

### 구성

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
| `samples/` | | 예제 사진·영상·소리(퍼블릭 도메인·CC0, 출처는 `samples/SOURCES.md`) |

## 모델과 라이선스

모델은 이 저장소에 들어 있지 않습니다.

- `hexagon/get-models.sh`는 Qualcomm AI Hub가 공개한 모델(`.tflite`)을 내려받습니다. 모델마다 라이선스(BSD-3-Clause, Apache-2.0, MIT)가 다르므로
  Hugging Face의 `qualcomm/<모델>` 페이지에서 확인하세요.
- `tools/get-models.sh`는 DEEPX 모델 저장소의 모델(`.dxnn`, `.onnx`)을 내려받습니다. DEEPX가 오픈소스 모델을 변환해 배포하는
  평가·개발용 모델이므로, 제품에 넣어 배포할 때는 원 모델의 라이선스를 확인하세요.

`hexagon/clip_tokenizer.py`와 `data/bpe_simple_vocab_16e6.txt.gz`는 OpenAI CLIP(MIT License)에서 가져왔습니다. 전문은 `data/CLIP_LICENSE`에 있습니다.

이 저장소의 코드 라이선스는 `LICENSE`를 따릅니다.
