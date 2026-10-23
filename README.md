# iq-9075-evk-labs

Qualcomm Dragonwing **IQ-9075 EVK** 한 대로 해 보는 엣지 AI 예제 모음입니다. ARM CPU가 영상 해독·전처리·후처리·서비스를 맡고,
NPU가 신경망 계산을 맡는 구성을 실습과 현장형 응용으로 익힙니다.

NPU는 두 가지를 다룹니다.

- **내장 NPU**: IQ-9075 프로세서 안의 Qualcomm Hexagon NPU(HTP 2개). 추가 장치 없이 바로 씁니다. → `hexagon/`
- **DX-M1**: 보드의 PCIe x4 슬롯에 연결하는 DEEPX DX-M1 NPU 모듈(M.2 모듈, PCIe Gen3 x4로 동작). → `labs/`, `apps/`, `tools/`

둘 중 하나만 있어도 되고, 둘 다 있으면 한 프로그램에서 함께 쓰는 예제도 돌릴 수 있습니다.
이 README의 명령과 수치는 2026년 10월 IQ-9075 EVK 실제 장비에서 확인한 것입니다.

## 목차

- [무엇을 해 볼 수 있나](#무엇을-해-볼-수-있나)
- [필요한 것](#필요한-것)
- [저장소 구성](#저장소-구성)
- [시작하기 A: 내장 NPU](#시작하기-a-내장-npu)
- [시작하기 B: DX-M1](#시작하기-b-dx-m1)
- [시작하기 C: 두 NPU 함께](#시작하기-c-두-npu-함께)
- [내장 NPU 예제](#내장-npu-예제)
- [DX-M1 예제](#dx-m1-예제)
- [측정·운영 도구](#측정운영-도구)
- [측정 결과 요약](#측정-결과-요약)
- [문제 해결](#문제-해결)
- [알아 둘 제약](#알아-둘-제약)
- [모델과 자료의 라이선스](#모델과-자료의-라이선스)

## 무엇을 해 볼 수 있나

예제는 모두 42개입니다. 내장 NPU용이 23개(실습 12, 응용 11), DX-M1용이 19개(실습 11, 응용 8)입니다.

| 해 볼 수 있는 것 | 예제 | 검증 장비에서 나온 결과 |
|---|---|---|
| CPU와 NPU 속도 비교 | `hexagon/labs/h03_cpu_vs_npu.py` | YOLOX 한 장: ARM CPU 8코어 57.5 ms, 내장 NPU 3.2 ms(18배) |
| 안전모·안전 조끼 점검 | `hexagon/apps/ppe_check.py`, `apps/ppe_two_npus.py` | 1080p 영상 38.5 FPS(내장 NPU), 두 NPU로 나누면 79 FPS |
| 구역 진입·체류 감시 | `hexagon/apps/zone_guard.py`, `apps/zone_guard.py` | 진입·체류 기록과 사진(얼굴은 흐리게), 69~71 FPS |
| 통행 인원·차량 세기 | `hexagon/apps/people_counter.py`, `apps/line_counter.py` | 방향별 1분 집계 CSV, 84~86 FPS |
| 영상 속 얼굴 가리기 | `hexagon/apps/privacy_blur.py`, `apps/privacy_blur.py` | 88~101 FPS |
| 학습 없이 사진 나누기 | `hexagon/apps/sort_photos.py` | 종류 이름만 주고 32장 중 31장 맞힘(CLIP) |
| 글자 읽기(OCR) | `hexagon/apps/read_text.py` | 경고 표지판의 영문을 읽음. 글자 찾기 장당 30 ms |
| 소리 감시 | `hexagon/apps/sound_watch.py` | 사이렌·경보·경적을 실시간의 342배 속도로 분류 |
| 사내 추론 서버 | `hexagon/apps/infer_server.py`, `apps/infer_server.py` | HTTP로 사진을 받아 JSON 응답. 검출 초당 276건(내장 NPU) |
| 여러 카메라 동시 처리 | `hexagon/apps/multi_camera.py`, `apps/multi_stream.py` | 1080p·24 FPS 카메라 8대(내장 NPU), 6대(DX-M1)를 640×360으로 분석해 실시간 |
| 두 NPU 동시에 | `labs/11_two_npus.py` | 각자 처리량의 93~98%를 유지해 합계 초당 800장 안팎 검출 |

FPS는 1080p 보행자 영상(`samples/videos/pedestrians.mp4`)을 파일에서 읽어 처리한 속도입니다.

## 필요한 것

| 항목 | 내용 |
|---|---|
| 보드 | Qualcomm Dragonwing IQ-9075 EVK. 출고 Ubuntu 24.04(커널 6.8.0-1080-qcom, 6.8.0-1084-qcom에서 확인) |
| DX-M1(선택) | DEEPX DX-M1 M.2 모듈. 드라이버 2.6.0, 런타임 DX-RT 3.4.2, 펌웨어 2.7.0 이상 |
| 네트워크 | 패키지·모델을 받을 인터넷 연결 |
| 저장 공간 | 내장 NPU 모델 약 920 MB, DX-M1 모델 약 480 MB, Python 가상환경 수백 MB |
| 작업용 PC | SSH 접속과 결과 그림·영상 보기용(선택) |

명령은 모두 보드에서 `ubuntu` 계정으로 실행합니다. 보드에 SSH로 접속하거나 모니터·키보드를 연결해 터미널을 엽니다.

## 저장소 구성

```text
iq-9075-evk-labs/
├── hexagon/              내장 NPU(Qualcomm Hexagon, LiteRT + QNN 델리게이트)
│   ├── labs/             실습 h01~h12
│   ├── apps/             응용 11개
│   ├── qnpu.py           공용 도구: 모델 실행(HTP 지정, 성능 모드, 여러 HTP 동시 실행), 후처리
│   ├── clip_tokenizer.py CLIP 글 토크나이저(OpenAI CLIP, MIT)
│   ├── get-models.sh     모델 내려받기(Qualcomm AI Hub 공개 모델 → ~/qai-models)
│   ├── bench.py, soak.py, ops.py, temps.sh   성능 표, 오래 돌리기, 모델 들여다보기, 온도
│   └── requirements.txt
├── labs/                 DX-M1 실습 01~11
├── apps/                 DX-M1 응용 8개(apps/infer_client.py는 두 서버 공용 호출 도구)
├── tools/                DX-M1 도구: 모델 내려받기(→ ~/npu-models), 성능 표, 온도, 서버 서비스 파일
├── labkit.py             공용 도구: 전처리, YOLO 후처리, 영상 입출력, 추적기, CPU 사용량 측정
├── requirements.txt      DX-M1 예제용 Python 패키지
├── data/                 ImageNet 분류 이름, CLIP 단어 사전
└── samples/              예제 사진·영상·소리(퍼블릭 도메인·CC0, 출처는 samples/SOURCES.md)
```

모델 파일은 저장소에 들어 있지 않습니다. 내려받기 스크립트가 홈 폴더의 `~/qai-models`, `~/npu-models`에 받습니다.

## 시작하기 A: 내장 NPU

추가 장치 없이 IQ-9075만으로 시작합니다. 모델 내려받기를 포함해 10분 정도 걸립니다.

**1. Qualcomm 패키지 설치**

IQ-9075의 Ubuntu에는 Qualcomm 패키지 저장소(PPA `ubuntu-qcom-iot/qcom-ppa`)가 처음부터 등록되어 있습니다.
목록만 갱신하고 설치된 패키지는 올리지 않습니다(`apt upgrade`는 하지 않아도 됩니다).

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends qairt-libs qairt-dsp-binaries qairt-tools tensorflow-lite-qcom-apps python3-venv
```

| 패키지 | 역할 |
|---|---|
| `qairt-libs` | Qualcomm AI Runtime 라이브러리(QNN HTP 백엔드) |
| `qairt-dsp-binaries` | NPU 쪽에서 동작하는 라이브러리 |
| `qairt-tools` | 점검·변환 도구(`qnn-platform-validator` 등) |
| `tensorflow-lite-qcom-apps` | LiteRT용 QNN 델리게이트 `/usr/lib/libQnnTFLiteDelegate.so` |

**2. NPU 동작 확인**

```bash
cd /tmp
qnn-platform-validator --backend dsp --testBackend | grep "Unit Test"
```

```text
Unit Test on the backend DSP: Passed.
  Unit Test         : Passed
```

`Passed`이면 정상입니다. 이 명령은 정상일 때도 마지막에 `Error in saving the results`를 내고 종료 코드 255로 끝나므로, 결과는 `Unit Test` 줄로 판단합니다.

**3. 저장소, Python 환경, 모델**

```bash
cd ~
git clone https://github.com/Gyurinet/iq-9075-evk-labs.git
python3 -m venv ~/qai-venv
~/qai-venv/bin/pip install -r ~/iq-9075-evk-labs/hexagon/requirements.txt
bash ~/iq-9075-evk-labs/hexagon/get-models.sh
```

`get-models.sh`는 예제에 쓰는 모델 15종을 Qualcomm AI Hub 공개 저장소에서 받습니다(약 920 MB, 계정 불필요).
일부만 받으려면 이름을 줍니다. 예: `bash hexagon/get-models.sh yolox gear_guard_net`

**4. 첫 실행**

```bash
cd ~/iq-9075-evk-labs
source ~/qai-venv/bin/activate
python3 hexagon/labs/h01_hello_npu.py
```

```text
모델 올리기: 0.79초
입력 : image_tensor [...] uint8 (0.003918204456567764, 0)
출력 : class_logits [...] uint8 (0.14131049811840057, 53)
1회: 2.80 ms, 출력 (1000,)
2회: 1.39 ms, 출력 (1000,)
...
```

처음 실행할 때는 NPU용 그래프를 만드느라 모델 올리기가 몇 초 걸리고, 이후에는 캐시(`~/.cache/qnn-delegate`)로 1초 안에 올라갑니다.
새 터미널을 열 때마다 `source ~/qai-venv/bin/activate`로 가상환경을 켭니다.

## 시작하기 B: DX-M1

DX-M1 모듈을 보드에 꽂고 전원을 켠 뒤 진행합니다. 15분 정도 걸립니다.

**1. 모듈 인식 확인**

```bash
lspci | grep -i deepx
```

```text
0001:01:00.0 Processing accelerators: DEEPX Co., Ltd. DX_M1 (rev 01)
```

아무것도 나오지 않으면 모듈 장착과 전원을 확인합니다.

**2. 드라이버·런타임 설치**

DEEPX가 GitHub에 배포하는 Debian 패키지 두 개를 받아 설치합니다. 패키지 목록이 오래되면 빌드 도구 설치가 `404 Not Found`로 실패하므로 먼저 갱신합니다.

```bash
sudo apt-get update
mkdir -p ~/Downloads
cd ~/Downloads
wget https://github.com/DEEPX-AI/dx_rt_npu_linux_driver/raw/refs/heads/main/release/2.6.0/dxrt-driver-dkms_2.6.0-2_all.deb
wget https://github.com/DEEPX-AI/dx_rt/raw/refs/heads/main/release/3.4.2/libdxrt-bin_3.4.2_arm64.deb
sha256sum *.deb
```

```text
6ab417956d2abd70c181c4aa0bae71bc60e0961ccf37eac9fc82d0e6a8314468  dxrt-driver-dkms_2.6.0-2_all.deb
161223058938c54efadc581b33499210194da58e55cd5ade332e692ca0d649c2  libdxrt-bin_3.4.2_arm64.deb
```

```bash
cd ~/Downloads
sudo apt-get install -y --no-install-recommends ./dxrt-driver-dkms_2.6.0-2_all.deb
sudo apt-get install -y ./libdxrt-bin_3.4.2_arm64.deb
lsmod | grep dx
systemctl is-active dxrt.service
```

`dxrt_driver`, `dx_dma` 두 줄과 `active`가 나오면 정상입니다. 드라이버는 DKMS로 설치되어 커널이 바뀌면 자동으로 다시 빌드됩니다.

**3. 펌웨어 확인과 업데이트**

DX-RT 3.4.2는 모듈 펌웨어가 2.7.0 이상이어야 모델을 실행합니다.

```bash
dxrt-cli -s | grep "FW version"
```

2.7.0보다 낮으면 DEEPX 펌웨어 저장소에서 M.2용 펌웨어를 받아 기록합니다. **기록하는 동안 전원을 끄지 마세요.** 모듈이 손상될 수 있습니다.

```bash
cd ~
git clone --depth 1 https://github.com/DEEPX-AI/dx_fw.git
cd ~/dx_fw
dxrt-cli -g m1/latest/mdot2/fw.bin
sudo systemctl stop dxrt.service
dxrt-cli -u m1/latest/mdot2/fw.bin
sudo systemctl start dxrt.service
dxrt-cli -s | grep "FW version"
```

**4. 저장소, Python 환경, 모델**

```bash
sudo apt-get install -y --no-install-recommends gstreamer1.0-plugins-bad python3-venv
cd ~
git clone https://github.com/Gyurinet/iq-9075-evk-labs.git
python3 -m venv ~/npu-venv
~/npu-venv/bin/pip install /usr/share/libdxrt-bin/python/dx_engine-3.4.2-cp312-cp312-manylinux_2_27_aarch64.whl
~/npu-venv/bin/pip install -r ~/iq-9075-evk-labs/requirements.txt
bash ~/iq-9075-evk-labs/tools/get-models.sh
```

`dx_engine`(DX-RT의 Python 패키지)은 런타임이 `/usr/share/libdxrt-bin/python/`에 넣어 둔 파일로 설치합니다. 시작하기 A에서 저장소를 이미 받았다면 `git clone` 줄은 건너뜁니다.
`get-models.sh`는 DEEPX 모델 저장소에서 모델 9개(`.dxnn`)와 CPU 비교용 원본 4개(`.onnx`)를 받습니다(약 480 MB).

**5. 첫 실행**

```bash
cd ~/iq-9075-evk-labs
source ~/npu-venv/bin/activate
python3 labs/01_hello_npu.py
```

```text
입력 : input.1 [1, 224, 224, 3] uint8
출력 : 495 [1, 1000] float32
1회: NPU 계산 2.00 ms, 전체 3.55 ms, 출력 (1, 1000)
2회: NPU 계산 2.00 ms, 전체 2.67 ms, 출력 (1, 1000)
...
```

출력 앞에 `[DXAPP]`나 `[DXRT]`로 시작하는 런타임 안내 줄이 함께 나올 수 있습니다.

## 시작하기 C: 두 NPU 함께

시작하기 A의 1단계(Qualcomm 패키지)와 시작하기 B를 마친 뒤, DX-M1용 가상환경에 LiteRT를 더하고 내장 NPU 모델 두 개를 받습니다.

```bash
~/npu-venv/bin/pip install ai-edge-litert==2.3.0
bash ~/iq-9075-evk-labs/hexagon/get-models.sh yolox gear_guard_net
cd ~/iq-9075-evk-labs
source ~/npu-venv/bin/activate
python3 labs/11_two_npus.py
```

```text
사진 1장을 계속 검출, 10초씩
                    DX-M1(YOLO26n)     내장 NPU(YOLOX)       합계      CPU
DX-M1만                  217.5 FPS           0.0 FPS    217.5    1.6코어
내장 NPU만                 0.0 FPS         609.4 FPS    609.4    2.0코어
둘 다 동시에             214.1 FPS         589.5 FPS    803.6    3.5코어
```

두 NPU는 따로 계산하고 ARM CPU와 메모리만 함께 쓰므로, 동시에 돌려도 각자의 처리량이 거의 그대로입니다.

## 내장 NPU 예제

예제는 결과 파일이 한곳에 모이도록 작업 폴더 `~/work`에서 실행합니다. 아래 명령은 새 터미널에 그대로 붙여 넣어 실행할 수 있습니다.

### 실습(h01~h12)

| 파일 | 배우는 것 |
|---|---|
| `h01_hello_npu.py` | 모델을 NPU에 올리고 실행하기, 입력·출력 텐서와 양자화 값(배율·영점) |
| `h02_classify.py` | 이미지 분류(ResNet50, ImageNet 1000종) |
| `h03_cpu_vs_npu.py` | 같은 모델 파일을 ARM CPU(XNNPACK)와 NPU(HTP 1개·2개, 성능 모드)에서 비교 |
| `h04_detect.py` | 객체 검출(YOLOX, COCO 80종). 상자 정리(NMS)는 ARM |
| `h05_pipeline.py` | 영상 처리 단계별 시간 재기, HTP 2개에 나눠 겹쳐 처리 |
| `h06_people.py` | 사람·얼굴·발 위치를 한 번에(FootTrackNet) |
| `h07_segment.py` | 도로 장면 분할(FFNet, Cityscapes 19종) |
| `h08_depth.py` | 사진 한 장으로 깊이 추정(Depth Anything V2, fp16) |
| `h09_quant.py` | 양자화 들여다보기: 8비트 텐서의 배율, 실수 모델과 결과 비교 |
| `h10_pose.py` | 두 단계 모델: 사람을 찾고(FootTrackNet) 사람마다 관절 17개(HRNet) |
| `h11_two_models.py` | 서로 다른 모델 두 개를 HTP 2개에 나눠 동시에 |
| `h12_super_res.py` | 초해상도(XLSR): 4배 확대를 보통 확대와 비교(PSNR) |

```bash
mkdir -p ~/work
cd ~/work
source ~/qai-venv/bin/activate
L=~/iq-9075-evk-labs
S=$L/samples
python3 $L/hexagon/labs/h02_classify.py $S/images/forklift.jpg $S/images/hammer.jpg
python3 $L/hexagon/labs/h03_cpu_vs_npu.py yolox
python3 $L/hexagon/labs/h04_detect.py $S/images/crossing.jpg -o detect.jpg
python3 $L/hexagon/labs/h05_pipeline.py $S/videos/crossing.mp4
python3 $L/hexagon/labs/h06_people.py $S/images/pedestrians.jpg -o people.jpg
python3 $L/hexagon/labs/h07_segment.py $S/images/roundabout.jpg -o segment.jpg
python3 $L/hexagon/labs/h08_depth.py $S/images/forklift.jpg -o depth.jpg
python3 $L/hexagon/labs/h09_quant.py $S/photos/*.jpg
python3 $L/hexagon/labs/h10_pose.py $S/ppe/road_crew.jpg -o pose.jpg
python3 $L/hexagon/labs/h11_two_models.py $S/videos/roundabout.mp4
python3 $L/hexagon/labs/h12_super_res.py $S/images/forklift.jpg -o super_res.jpg
```

`h03_cpu_vs_npu.py`는 `resnet50`, `mobilenet_v2`, `yolox`, `resnet50-float`(실수 모델을 NPU에서 fp16으로) 중 하나를 받습니다.

### 응용

응용은 모두 같은 준비 줄로 시작합니다. 아래 명령들은 이 준비 줄을 실행한 터미널에서 이어서 실행합니다.

```bash
mkdir -p ~/work
cd ~/work
source ~/qai-venv/bin/activate
L=~/iq-9075-evk-labs
S=$L/samples
```

**안전 장비 점검** `ppe_check.py`: 사람을 찾고 사람마다 안전모·안전 조끼 착용을 판정합니다(두 모델 잇기). 사진 폴더를 주면 결과 사진과 `report.csv`, 영상을 주면 미착용 기록 `events.csv`를 남깁니다. 얼굴은 흐리게 가려 저장합니다.

```bash
python3 $L/hexagon/apps/ppe_check.py $S/ppe -o ppe_out
python3 $L/hexagon/apps/ppe_check.py $S/videos/pedestrians.mp4 --events events.csv
```

**통행 인원 세기** `people_counter.py`: 사람을 추적해 가상의 선을 넘는 인원을 방향별·1분 단위로 셉니다(`people.csv`). 선은 `--line x1,y1,x2,y2`(화면 폭·높이에 대한 비율 0~1)로 정합니다.

```bash
python3 $L/hexagon/apps/people_counter.py $S/videos/pedestrians.mp4 --save counted.mp4
```

**구역 감시** `zone_guard.py`: 정한 구역에 들어오거나 오래 머문 사람을 기록하고(`events.csv`) 그 순간의 사진을 남깁니다(`snapshots/`). 구역은 `--zone`, 체류 기준은 `--dwell 초`로 정합니다.

```bash
python3 $L/hexagon/apps/zone_guard.py $S/videos/pedestrians.mp4 --save zone.mp4
```

**얼굴 가리기** `privacy_blur.py`: 영상을 외부에 넘기기 전에 얼굴을 흐리게 가립니다. `--show`를 붙이면 찾은 얼굴에 상자를 그려 확인할 수 있습니다.

```bash
python3 $L/hexagon/apps/privacy_blur.py $S/images/pedestrians.jpg -o blurred.jpg
python3 $L/hexagon/apps/privacy_blur.py $S/videos/pedestrians.mp4 --save blurred.mp4
```

**사람 오려내기** `cutout.py`: 화소마다 사람인지 판정해(DeepLabV3+) 배경을 흐리게(`blur-bg`) 하거나 한 색으로(`remove-bg`) 바꾸고, 사람 전체를 가리기도 합니다(`hide`). `--cls car`처럼 다른 종류(VOC 20종)도 고를 수 있습니다.

```bash
python3 $L/hexagon/apps/cutout.py $S/ppe/loading.jpg -o blur.jpg --mode blur-bg
python3 $L/hexagon/apps/cutout.py $S/videos/pedestrians.mp4 --save hide.mp4 --mode hide
```

**사진 검색** `photo_search.py`: CLIP으로 사진 폴더를 색인해 두고 영어 문장이나 사진으로 찾습니다.

```bash
python3 $L/hexagon/apps/photo_search.py index $S/photos
python3 $L/hexagon/apps/photo_search.py text "an excavator with a grapple"
python3 $L/hexagon/apps/photo_search.py image $S/images/forklift.jpg
```

**학습 없이 사진 나누기** `sort_photos.py`: 종류 이름(영어)만 주면 사진마다 가장 맞는 종류를 고르고, `--out` 폴더에 종류별로 복사합니다(원본은 그대로).

```bash
C="forklift,excavator,tractor,hard hat,padlock,power drill,bicycle,cardboard box"
python3 $L/hexagon/apps/sort_photos.py $S/photos --classes "$C" --out sorted
```

```text
파일 이름이 정답인 32장 중 31장 맞음(97%)
```

**글자 읽기(OCR)** `read_text.py`: 글자가 있는 곳은 NPU가 찾고(EasyOCR 검출기), 글자 읽기는 ARM CPU가 합니다. 인식기는 NPU에 올라가지 않습니다([알아 둘 제약](#알아-둘-제약)). 인쇄된 영문·숫자에 알맞고 한글은 읽지 못합니다.

```bash
python3 $L/hexagon/apps/read_text.py $S/signs/*.jpg -o text_out
```

```text
high_voltage.jpg             글자 상자 3개, 찾기 31 ms, 읽기 399 ms | DANGER HIGH VOLTAGE
pole_sign.jpg                글자 상자 3개, 찾기 24 ms, 읽기 379 ms | 'Danger High Voltage
```

**소리 감시** `sound_watch.py`: 녹음을 0.96초씩 잘라 소리 종류(AudioSet 521종, YAMNet)를 알아내고, 지켜볼 소리(`--watch`)가 나면 기록합니다. 입력은 16 kHz WAV입니다(다른 형식은 `ffmpeg -i 원본 -ac 1 -ar 16000 결과.wav`로 바꿉니다).

```bash
python3 $L/hexagon/apps/sound_watch.py $S/sounds/*.wav --csv sounds.csv
```

**사내 추론 서버** `infer_server.py`: 사내망의 PC·프로그램이 사진을 HTTP로 보내면 NPU로 분석해 JSON으로 돌려줍니다. 터미널 하나에서 서버를 켭니다(`Ctrl+C`로 끔).

```bash
cd ~/iq-9075-evk-labs
source ~/qai-venv/bin/activate
python3 hexagon/apps/infer_server.py --port 8080
```

| 주소 | 하는 일 |
|---|---|
| `GET /health` | 서버 상태와 모델 |
| `POST /detect?conf=0.4` | 사진 → 검출한 물체 목록(YOLOX, COCO 80종) |
| `POST /classify?top=5` | 사진 → 분류 상위 n개(ResNet50) |
| `POST /people` | 사진 → 사람·얼굴 수와 사람 상자 |
| `POST /tags?labels=a,b,c` | 사진 → 주어진 영어 낱말(5개까지)별 점수(CLIP) |

다른 터미널이나 사내 PC에서 호출합니다(사내 PC에서는 `127.0.0.1` 대신 장비 IP). `apps/infer_client.py`는 표준 라이브러리만 쓰므로 어느 PC의 Python에서나 돌고, `--bench`로 부하 시험도 합니다.

```bash
cd ~/iq-9075-evk-labs
curl -s --data-binary @samples/images/pedestrians.jpg http://127.0.0.1:8080/people
python3 apps/infer_client.py "http://127.0.0.1:8080/tags?labels=forklift,tractor,truck" samples/images/forklift.jpg
python3 apps/infer_client.py http://127.0.0.1:8080/detect samples/images/crossing.jpg --bench 300 --clients 8
```

**여러 카메라 동시 처리** `multi_camera.py`: 카메라 N대의 프레임을 실시간 속도로 받아 검출합니다. 영상 파일은 `--streams N`으로 N대처럼 쓰고, 실제 카메라는 `rtsp://` 주소를 여러 개 적습니다. 처리가 밀리면 오래된 프레임을 버립니다(실시간 우선).

```bash
python3 $L/hexagon/apps/multi_camera.py $S/videos/crossing.mp4 --streams 8 --seconds 15
```

## DX-M1 예제

내장 NPU 예제와 같은 방식으로 `~/work`에서 실행하고, 가상환경은 `~/npu-venv`를 씁니다.

### 실습(01~11)

| 파일 | 배우는 것 |
|---|---|
| `01_hello_npu.py` | 모델을 NPU에 올리고 실행하기, 입력·출력 텐서 |
| `02_classify.py` | 이미지 분류(ResNet50) |
| `03_cpu_vs_npu.py` | 같은 모델을 ARM CPU(ONNX Runtime)와 NPU에서 비교(`resnet50`, `mobilenetv2`, `yolo11n`, `yolo26n`) |
| `04_pipeline.py` | 영상 검출 단계별 시간 재기, 비동기 요청으로 겹쳐 처리 |
| `05_detect.py` | 객체 검출(YOLO11n, YOLO26n) |
| `06_segment.py`, `07_pose.py`, `08_depth.py` | 인스턴스 분할, 자세 추정(관절 17개), 깊이 추정 |
| `09_npu_cores.py` | NPU 코어 3개 나눠 쓰기: 급한 작업의 응답 시간 지키기 |
| `10_video_decode.py` | 영상 해독: ARM 소프트웨어와 하드웨어 디코더 비교([알아 둘 제약](#알아-둘-제약)) |
| `11_two_npus.py` | DX-M1과 내장 NPU 동시에([시작하기 C](#시작하기-c-두-npu-함께)) |

```bash
mkdir -p ~/work
cd ~/work
source ~/npu-venv/bin/activate
L=~/iq-9075-evk-labs
S=$L/samples
python3 $L/labs/02_classify.py $S/images/forklift.jpg $S/images/hammer.jpg
python3 $L/labs/03_cpu_vs_npu.py yolo26n
python3 $L/labs/04_pipeline.py $S/videos/crossing.mp4
python3 $L/labs/05_detect.py $S/images/crossing.jpg --model yolo26n -o detect.jpg
python3 $L/labs/06_segment.py $S/images/crossing.jpg -o segment.jpg
python3 $L/labs/07_pose.py $S/images/pedestrians.jpg -o pose.jpg
python3 $L/labs/08_depth.py $S/images/roundabout.jpg -o depth.jpg
```

`09_npu_cores.py`는 무거운 일로 DeepLabV3+ 분할 모델을 쓰므로, 이 모델을 먼저 따로 받습니다(`get-models.sh`에는 들어 있지 않습니다).

```bash
cd ~/npu-models
wget https://sdk.deepx.ai/modelzoo/dxnn/2_4_0/deeplabv3plus_mobilenetv1_512x512.dxnn
cd ~/iq-9075-evk-labs
source ~/npu-venv/bin/activate
python3 labs/09_npu_cores.py 10
```

```text
분류만(기준)                        분류 응답 중앙값   8.79 ms, 95%  11.24 ms | 분할 없음
함께, 코어 0·1·2 함께 쓰기             분류 응답 중앙값  17.91 ms, 95%  23.36 ms | 분할  223.5 FPS
함께, 분류 코어 0 / 분할 코어 1·2        분류 응답 중앙값  10.08 ms, 95%  12.31 ms | 분할  165.7 FPS
```

급한 일(분류)에 코어 0을, 무거운 일(분할)에 코어 1·2를 나눠 주면 분류 응답 시간이 거의 그대로 돌아옵니다. 응답 시간은 CPU 주파수 정책에 따라 달라지므로, 비교할 때는 `performance` 정책에서 재면 흔들림이 줄어듭니다.

### 응용

```bash
mkdir -p ~/work
cd ~/work
source ~/npu-venv/bin/activate
L=~/iq-9075-evk-labs
S=$L/samples
```

| 응용 | 하는 일 | 실행 예 |
|---|---|---|
| `line_counter.py` | 가상의 선을 지나는 사람·자전거·차량을 방향별로 세기, 1분 집계 CSV | `python3 $L/apps/line_counter.py $S/videos/pedestrians.mp4 --save counted.mp4` |
| `zone_guard.py` | 구역 진입·체류 기록과 사진 | `python3 $L/apps/zone_guard.py $S/videos/pedestrians.mp4 --save zone.mp4` |
| `privacy_blur.py` | 얼굴 가리기(YOLOv5s-face) | `python3 $L/apps/privacy_blur.py $S/videos/pedestrians.mp4 --save blurred.mp4` |
| `photo_search.py` | 비슷한 사진·중복 사진 찾기(CLIP 이미지 인코더) | `python3 $L/apps/photo_search.py index $S/photos` 다음 `query <사진>`, `dups` |
| `multi_stream.py` | 여러 카메라 동시 처리 | `python3 $L/apps/multi_stream.py $S/videos/crossing.mp4 --streams 6 --seconds 15` |
| `infer_server.py` | 사내 추론 서버(`/health`, `/detect`, `/classify`) | `python3 $L/apps/infer_server.py --port 8080` |
| `infer_client.py` | 서버 호출과 부하 시험(두 서버 공용) | `python3 $L/apps/infer_client.py http://<장비 IP>:8080/detect <사진>` |
| `ppe_two_npus.py` | 두 NPU로 안전 장비 점검: DX-M1이 사람·얼굴, 내장 NPU가 안전모·조끼([시작하기 C](#시작하기-c-두-npu-함께) 준비 필요) | `python3 $L/apps/ppe_two_npus.py $S/ppe -o ppe_out` |

DX-M1 추론 서버를 부팅할 때 자동으로 켜려면 `tools/arm-npu-server.service`를 씁니다(계정 `ubuntu`, 저장소 `~/iq-9075-evk-labs`, 가상환경 `~/npu-venv` 기준. 다르면 파일 안의 `User`, `WorkingDirectory`, `ExecStart`를 고칩니다).

```bash
sudo cp ~/iq-9075-evk-labs/tools/arm-npu-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now arm-npu-server
```

## 측정·운영 도구

저장소 폴더(`~/iq-9075-evk-labs`)에서 해당 가상환경을 켜고 실행합니다.

| 도구 | 하는 일 |
|---|---|
| `python3 hexagon/bench.py [모델 …]` | 내장 NPU 모델별 ARM CPU 8코어와 NPU(기본·최고 성능 모드)의 한 장 처리 시간(전체 3분 정도) |
| `python3 hexagon/ops.py <모델>` | 모델의 입력·출력, 연산 종류별 개수, 내장 NPU에 올라가는지 |
| `python3 hexagon/soak.py --minutes 30` | HTP 2개를 최고 성능 모드로 계속 쓰며 처리량과 온도 기록 |
| `bash hexagon/temps.sh` | 내장 NPU(HTP 0·1), ARM CPU, GPU 온도 |
| `bash tools/bench.sh [모델.dxnn …]` | DX-M1 모델별 NPU 계산 시간, 한 장씩 FPS, 최대 처리량 |
| `bash tools/temps.sh` | DX-M1 NPU 코어 온도·클록과 ARM CPU 최고 온도 |

새 모델을 들일 때는 먼저 `hexagon/ops.py`로 NPU에 올라가는지, `hexagon/bench.py`로 얼마나 빠른지 확인합니다.

## 측정 결과 요약

내장 NPU, 한 장 처리 시간(`hexagon/bench.py`). 배율은 CPU 8코어 시간 ÷ NPU 최고 성능 모드 시간입니다.

| 모델 | 입력 | 형식 | ARM CPU 8코어 | NPU 기본 | NPU 최고 성능 | 배율 |
|---|---|---|---:|---:|---:|---:|
| ResNet50 | 224×224 | int8 | 11.7 ms | 2.0 ms | 0.9 ms | 13배 |
| YOLOX | 640×640 | int8 | 57.5 ms | 6.1 ms | 3.2 ms | 18배 |
| FootTrackNet(사람·얼굴) | 480×640 | int8 | 29.7 ms | 4.8 ms | 2.8 ms | 11배 |
| HRNet(자세) | 256×192 | int8 | 25.4 ms | 2.8 ms | 1.2 ms | 21배 |
| DeepLabV3+ | 520×520 | int8 | 68.6 ms | 9.7 ms | 5.3 ms | 13배 |
| FFNet-40S | 1024×2048 | int8 | 135.7 ms | 18.0 ms | 10.9 ms | 12배 |
| EasyOCR 검출기 | 608×800 | int8 | 340.9 ms | 20.6 ms | 12.8 ms | 27배 |
| Depth Anything V2 | 518×518 | fp16 | 557.7 ms | 87.6 ms | 55.3 ms | 10배 |
| CLIP(사진 1장 + 문장 5개) | 224×224 | fp16 | 306.2 ms | 59.7 ms | 37.6 ms | 8배 |

DX-M1, `tools/bench.sh`. 최대 처리량은 요청을 겹쳐 NPU 코어 3개를 모두 쓸 때이며, 모델의 CPU 작업까지 포함합니다.

| 모델 | NPU 계산 | 최대 처리량 |
|---|---:|---:|
| ResNet50 224×224 | 2.0 ms | 1,080 FPS |
| YOLO26n 640×640 | 7.2 ms | 247 FPS |
| YOLO11n 640×640 | 5.2 ms | 89 FPS |
| YOLOv5s-face 640×640 | 4.6 ms | 293 FPS |
| CLIP ViT-B/32 이미지 256×256 | 11.5 ms | 186 FPS |

측정값은 CPU 주파수 정책, 다른 프로그램의 CPU 사용, 직전 NPU 부하에 따라 10~30% 달라질 수 있습니다.
내장 NPU는 HTP 2개를 최고 성능 모드로 5분 동안 계속 써도(YOLOX 초당 610장) 처리량이 떨어지지 않았고, NPU 온도는 68°C 근처에서 머물렀습니다(실내, 받은 그대로의 방열 구성).

## 문제 해결

| 증상 | 원인 | 조치 |
|---|---|---|
| `ModuleNotFoundError: No module named 'numpy'` 또는 `'dx_engine'` | 가상환경을 켜지 않음 | `source ~/qai-venv/bin/activate`(내장 NPU), `source ~/npu-venv/bin/activate`(DX-M1) |
| `OSError: libQnnTFLiteDelegate.so: cannot open shared object file` | QNN 델리게이트 미설치 | `tensorflow-lite-qcom-apps` 설치 |
| `모델이 없습니다: …` | 모델을 받지 않음 | `bash hexagon/get-models.sh` 또는 `bash tools/get-models.sh` |
| 내장 NPU에서 처음 실행할 때 모델 올리기가 몇 초 걸림 | NPU용 그래프를 만드는 중 | 정상. 두 번째부터 캐시로 빨라짐 |
| 내장 NPU 모델 준비 중 `RuntimeError` | NPU가 맡지 못하는 연산(반복문 `WHILE` 등) | `python3 hexagon/ops.py <모델>`로 확인하고 ARM CPU로 실행 |
| 원인을 보려는데 내장 NPU 기록이 안 보임 | `qnpu.py`가 델리게이트 기록을 숨김 | `QNN_VERBOSE=1 python3 …` |
| DX-M1 드라이버 설치 중 `404 Not Found` | 패키지 목록이 오래됨 | `sudo apt-get update` 후 다시 설치 |
| `The current firmware version is 2.5.1` | DX-M1 펌웨어가 낮음 | [시작하기 B](#시작하기-b-dx-m1)의 펌웨어 업데이트 |
| `dxrt service is not running` | NPU 관리 서비스 멈춤, 또는 모듈이 안 보임 | `systemctl status dxrt.service`, `lspci` 확인 |
| `no element "h264parse"` | GStreamer `plugins-bad` 없음 | `sudo apt-get install gstreamer1.0-plugins-bad` |
| 서버 시작 시 `Address already in use` | 서비스로 등록한 서버가 이미 8080 사용 | `sudo systemctl stop arm-npu-server` 또는 `--port` 변경 |
| 장비가 스스로 다시 켜짐 | 하드웨어 영상 코덱 문제 | 하드웨어 코덱(`--hw`)을 쓰지 않는다. `journalctl -b -1 -k`로 직전 기록 확인 |
| 겹쳐 처리해도 처리량이 늘지 않음 | NPU가 아닌 단계(영상 해독, 저장)가 병목 | `h05_pipeline.py`, `04_pipeline.py`로 단계별 시간 확인 |

## 알아 둘 제약

- **NPU에 올라가지 않는 모델이 있습니다.** EasyOCR 인식기는 반복문(`WHILE`)과 크기가 바뀌는 텐서가 있어 내장 NPU에 올라가지 않으므로 ARM CPU로 실행합니다(단어당 약 0.12초).
- **8비트 모델이 크게 틀리는 경우가 있습니다.** YAMNet 8비트 모델은 사이렌을 음악으로 답하는 등 결과가 틀어져, 실수 모델을 NPU에서 fp16으로 씁니다. 새 모델은 `h09_quant.py`처럼 실수 모델 결과와 비교합니다.
- **GPU는 쓰지 않습니다.** 검증한 Ubuntu에서는 OpenCL을 쓸 수 없어(`/dev/kgsl` 없음) QNN GPU 백엔드가 `Failed to retrieve OpenCL platform ID`로 실패합니다.
- **하드웨어 영상 코덱은 기본으로 쓰지 않습니다.** 검증 중 하드웨어 코덱(`v4l2h264dec`·`v4l2h264enc`)을 여러 개 열자 장비가 스스로 다시 켜진 일이 세 번 있었고, 그 뒤 DX-M1이 보이지 않아 전원을 다시 넣어야 했습니다. 예제는 ARM 소프트웨어 해독(OpenCV)이 기본이며, 하드웨어 코덱은 `--hw`로 고를 때만 씁니다.
- **내장 NPU의 그래프 캐시는 모델 구조로만 구분됩니다.** QNN 델리게이트에 `cache_dir`만 주면, 구조가 같고 가중치만 다른 모델(다시 학습·보정한 모델)이 먼저 만든 캐시를 그대로 써서 결과가 틀립니다. `qnpu.py`는 모델 파일의 경로·크기·수정 시각으로 `model_token`을 정해 파일마다 캐시를 따로 둡니다. 직접 델리게이트를 쓸 때도 `model_token`을 함께 줍니다.
- **내장 NPU의 `invoke()`는 끝날 때까지 기다리는 방식입니다.** 겹쳐 처리하려면 `qnpu.Pool`처럼 모델을 HTP마다 하나씩 올리고 스레드로 돌립니다. 최고 성능 모드는 HTP마다 CPU 코어 하나를 씁니다.

## 모델과 자료의 라이선스

모델은 이 저장소에 들어 있지 않고, 내려받기 스크립트가 공개 배포처에서 받습니다. 제품에 넣어 배포할 때는 각 원 모델의 라이선스를 확인하세요.

| 이름(예제에서) | 배포처의 모델 | 원 모델 | 라이선스 |
|---|---|---|---|
| `resnet50`, `resnet50-float` | AI Hub `resnet50` | torchvision ResNet50 | BSD-3-Clause |
| `mobilenet_v2` | AI Hub `mobilenet_v2` | pytorch-mobilenet-v2 | Apache-2.0 |
| `yolox` | AI Hub `yolox` | Megvii YOLOX | Apache-2.0 |
| `people` | AI Hub `foot_track_net` | Qualcomm FootTrackNet | BSD-3-Clause |
| `ppe` | AI Hub `gear_guard_net` | Qualcomm GearGuardNet | BSD-3-Clause |
| `ffnet` | AI Hub `ffnet_40s` | Qualcomm AI Research FFNet | BSD-3-Clause |
| `depth` | AI Hub `depth_anything_v2` | Depth Anything V2 | MIT |
| `clip` | AI Hub `openai_clip` | OpenAI CLIP | MIT |
| `pose` | AI Hub `hrnet_pose` | HRNet | MIT |
| `deeplab` | AI Hub `deeplabv3_plus_mobilenet` | DeepLabV3+ | MIT |
| `sr` | AI Hub `xlsr` | XLSR(AIMET Model Zoo) | BSD-3-Clause |
| `sound` | AI Hub `yamnet` | YAMNet | MIT |
| `ocr-det`, `ocr-rec` | AI Hub `easyocr` | JaidedAI EasyOCR | Apache-2.0 |
| DX-M1 모델(`.dxnn`) | DEEPX 모델 저장소 | ResNet50, MobileNetV2, YOLO11n(검출·분할·자세), YOLO26n, YOLOv5s-face, Depth Anything V2, CLIP | 평가·개발용 배포. 원 모델의 라이선스를 따름 |

- 내장 NPU 모델은 Qualcomm AI Hub 공개 모델 v0.65.0이며, 라이선스는 [qualcomm/ai-hub-models](https://github.com/qualcomm/ai-hub-models)의 모델 정보에 있습니다.
- `hexagon/clip_tokenizer.py`와 `data/bpe_simple_vocab_16e6.txt.gz`는 [OpenAI CLIP](https://github.com/openai/CLIP)(MIT License)에서 가져왔고, 전문은 `data/CLIP_LICENSE`에 있습니다.
- `samples/`의 사진·영상·소리는 Wikimedia Commons의 퍼블릭 도메인·CC0 자료이며, 파일마다 원본과 작성자를 `samples/SOURCES.md`에 적었습니다.

이 저장소의 코드 라이선스는 [`LICENSE`](LICENSE)를 따릅니다. 교육기관의 강의·실습과 기업·개인의 사내 평가·개발·시험에 쓰고 고칠 수 있으며,
판매하거나 상용 제품·서비스에 넣어 배포하려면 규리넷 주식회사의 서면 허가가 필요합니다.
