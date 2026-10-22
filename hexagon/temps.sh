#!/bin/bash
# IQ-9075의 내장 NPU(HTP 0·1), ARM CPU, GPU 온도 중 가장 높은 값
#   bash hexagon/temps.sh
#   watch -n 10 bash hexagon/temps.sh
max_of() {
  m=0
  for z in /sys/class/thermal/thermal_zone*; do
    case $(cat "$z/type") in
      $1) t=$(cat "$z/temp"); [ "$t" -gt "$m" ] && m=$t ;;
    esac
  done
  echo $((m / 1000))
}
echo "NPU HTP0 $(max_of 'nsp-0-*')'C, HTP1 $(max_of 'nsp-1-*')'C, ARM CPU $(max_of 'cpu-*')'C, GPU $(max_of 'gpuss-*')'C"
