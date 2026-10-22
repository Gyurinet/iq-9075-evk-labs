#!/bin/bash
# 실습에 쓰는 모델을 DEEPX 모델 저장소에서 내려받는다(~/npu-models, NPU_MODELS로 바꿀 수 있다)
#   bash tools/get-models.sh
# 모델은 DEEPX가 오픈소스 모델을 변환해 배포하는 것으로, 평가·개발용이다(상업용 배포는 원 모델의 라이선스를 따른다).
set -e
DIR=${NPU_MODELS:-$HOME/npu-models}
BASE=https://sdk.deepx.ai/modelzoo
mkdir -p "$DIR"
cd "$DIR"

# NPU 모델(.dxnn)
DXNN="resnet50_224x224 mobilenetv2_224x224 yolo11-n_640x640 yolo26-n_640x640 yolo11-n-seg_640x640
yolo11-n-pose_640x640 depthanythingv2-vits_224x224 yolov5-s-face_640x640 clip-img_vit-b32_256x256_datacomp-s34b-b86k"
# 같은 모델의 원본(.onnx): ARM CPU와 비교할 때 쓴다
ONNX="resnet50_224x224 mobilenetv2_224x224 yolo11-n_640x640 yolo26-n_640x640"

for m in $DXNN; do
  [ -s "$m.dxnn" ] || { echo "받는 중: $m.dxnn"; wget -q "$BASE/dxnn/2_4_0/$m.dxnn" -O "$m.dxnn.part" && mv "$m.dxnn.part" "$m.dxnn"; }
done
for m in $ONNX; do
  [ -s "$m.onnx" ] || { echo "받는 중: $m.onnx"; wget -q "$BASE/onnx/$m.onnx" -O "$m.onnx.part" && mv "$m.onnx.part" "$m.onnx"; }
done
echo "모델 폴더: $DIR"
ls -lh --time-style=+ "$DIR" | awk 'NR > 1 {print "  " $5 "\t" $NF}'
