#!/bin/bash
# 모델별 NPU 성능 표: NPU 계산 시간, 한 장씩 처리(FPS), 최대 처리량(FPS, 모델 안 CPU 작업 포함)
#   bash tools/bench.sh [모델.dxnn …]      (지정하지 않으면 ~/npu-models의 모든 .dxnn)
DIR=${NPU_MODELS:-$HOME/npu-models}
models=("$@")
[ ${#models[@]} -eq 0 ] && models=("$DIR"/*.dxnn)
printf "%-48s %10s %10s %10s\n" "모델" "NPU(ms)" "한장씩" "최대"
for m in "${models[@]}"; do
  single=$(dxrun -m "$m" -s -l 30 -w 5 -v --use-ort 2>&1)
  npu=$(echo "$single" | grep "NPU Processing Time" | tail -1 | awk '{print $6}')
  fps1=$(echo "$single" | grep -E "^\s+- FPS" | tail -1 | awk '{print $4}')
  fpsmax=$(dxrun -m "$m" -t 5 -w 10 --use-ort 2>&1 | grep -E "^\s+- FPS" | tail -1 | awk '{print $4}')
  printf "%-48s %10s %10s %10s\n" "$(basename "$m" .dxnn)" "$npu" "$fps1" "$fpsmax"
done
