#!/bin/bash
# IQ-9075 내장 NPU용 모델을 Qualcomm AI Hub 공개 저장소에서 내려받는다(~/qai-models, QAI_MODELS로 바꿀 수 있다)
#   bash hexagon/get-models.sh
# 모델의 라이선스는 Hugging Face의 qualcomm/<모델> 페이지에 있다(BSD-3-Clause, Apache-2.0, MIT).
set -e
DIR=${QAI_MODELS:-$HOME/qai-models}
BASE=https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models
VER=v0.65.0
mkdir -p "$DIR"
cd "$DIR"
# 모델 이름:정밀도. w8a8 = 8비트 정수(양자화), float = 실수(NPU에서는 fp16으로 실행)
MODELS="resnet50:w8a8 resnet50:float mobilenet_v2:w8a8 yolox:w8a8 foot_track_net:w8a8
gear_guard_net:w8a8 ffnet_40s:w8a8 depth_anything_v2:float openai_clip:float"
for spec in $MODELS; do
  m=${spec%%:*}
  p=${spec##*:}
  [ -d "$m-tflite-$p" ] && continue
  echo "받는 중: $m ($p)"
  wget -q "$BASE/$m/releases/$VER/$m-tflite-$p.zip"
  unzip -q -o "$m-tflite-$p.zip"
  rm "$m-tflite-$p.zip"
done
echo "모델 폴더: $DIR"
du -sh "$DIR"/*-tflite-* | sed "s|$DIR/||"
