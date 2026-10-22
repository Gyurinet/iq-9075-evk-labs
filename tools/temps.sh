#!/bin/bash
# NPU 코어 온도·클록과 ARM CPU의 가장 높은 온도를 보여 준다
#   bash tools/temps.sh          (한 번)
#   watch -n 10 bash tools/temps.sh   (10초마다)
dxrt-cli -s | grep "^NPU"
max=0
for z in /sys/class/thermal/thermal_zone*; do
  case $(cat "$z/type") in
    cpu*) t=$(cat "$z/temp"); [ "$t" -gt "$max" ] && max=$t ;;
  esac
done
echo "ARM CPU 최고 온도: $((max / 1000))'C"
