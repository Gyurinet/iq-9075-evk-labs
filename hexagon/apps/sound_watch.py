# 소리 감시: 녹음(WAV)을 0.96초씩 잘라 소리 종류(AudioSet 521종)를 NPU로 알아내고,
# 지켜볼 소리가 나면 기록한다
#   python3 hexagon/apps/sound_watch.py samples/sounds/*.wav
#   --watch "Siren,Alarm,Glass,Bark" : 지켜볼 소리(이름 일부), --th 0.3 : 기록할 점수
#   입력: 16 kHz, 16비트 WAV(다른 형식은 ffmpeg -i 원본 -ac 1 -ar 16000 결과.wav 로 바꾼다)
import argparse
import csv
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qnpu import Model, labels

ap = argparse.ArgumentParser()
ap.add_argument("wavs", nargs="+")
ap.add_argument("--watch", default="Siren,Alarm,Glass,Bark,Chainsaw,Drill,Engine,Car horn")
ap.add_argument("--th", type=float, default=0.3)
ap.add_argument("--csv", default="sounds.csv")
a = ap.parse_args()

SR, WIN, HOP, NFFT = 16000, 400, 160, 512  # 16 kHz, 25 ms 창, 10 ms 간격
PATCH = 96  # 0.96초 = 10 ms × 96


def mel_matrix(n_mel=64, lo=125.0, hi=7500.0):
    """FFT 칸(257개) → 멜 칸(64개) 가중치. 사람 귀처럼 낮은 소리를 촘촘히 나눈다"""
    mel = lambda f: 1127.0 * np.log1p(f / 700.0)  # noqa: E731
    bins = mel(np.linspace(0, SR / 2, NFFT // 2 + 1))
    edges = np.linspace(mel(lo), mel(hi), n_mel + 2)
    w = np.zeros((len(bins), n_mel), np.float32)
    for i in range(n_mel):
        lo_, mid, hi_ = edges[i : i + 3]
        w[:, i] = np.maximum(
            0, np.minimum((bins - lo_) / (mid - lo_), (hi_ - bins) / (hi_ - mid))
        )
    w[0] = 0
    return w


MEL = mel_matrix()
HANN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)


def log_mel(x):
    """소리(−1~1) → 로그 멜 스펙트로그램 [프레임 수, 64]. 이 계산은 ARM이 한다"""
    frames = np.lib.stride_tricks.sliding_window_view(x, WIN)[::HOP]
    spec = np.abs(np.fft.rfft(frames * HANN, NFFT))
    return np.log(spec @ MEL + 0.01).astype(np.float32)


def read_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2, (
            "16 kHz, 16비트 WAV가 필요합니다"
        )
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, w.getnchannels())
    return x.mean(1) / 32768.0


names = labels("sound")
watch = [w.strip().lower() for w in a.watch.split(",")]
watched = [i for i, n in enumerate(names) if any(w in n.lower() for w in watch)]
m = Model("sound")
rows, npu_ms, audio_s, t0 = [], [], 0.0, time.perf_counter()
for path in a.wavs:
    x = read_wav(path)
    audio_s += len(x) / SR
    spec = log_mel(x)
    found, tops = {}, {}
    for start in range(0, max(1, len(spec) - PATCH + 1), PATCH // 2):  # 0.48초씩 밀며
        patch = spec[start : start + PATCH]
        if len(patch) < PATCH:
            patch = np.pad(
                patch, ((0, PATCH - len(patch)), (0, 0)), constant_values=np.log(0.01)
            )
        logits = m.run(patch[None])["class_scores"]  # [521]
        scores = 1 / (
            1 + np.exp(-np.clip(logits, -30, 30))
        )  # 소리마다 0~1(여럿이 함께 높을 수 있다)
        npu_ms.append(m.last_ms)
        best = names[int(np.argmax(scores))]
        tops[best] = tops.get(best, 0) + 1  # 조각마다 1위인 소리를 센다
        hit = [i for i in watched if scores[i] >= a.th]
        for i in hit:
            found[names[i]] = max(found.get(names[i], 0), float(scores[i]))
            rows.append(
                [Path(path).name, f"{start * HOP / SR:.2f}", names[i], f"{scores[i]:.2f}"]
            )
    top = ", ".join(f"{k} {v:.2f}" for k, v in sorted(found.items(), key=lambda kv: -kv[1])[:3])
    most = max(tops, key=tops.get)
    name = Path(path).name
    print(f"{name:<20} {len(x) / SR:5.1f}초 | 가장 많은 소리: {most} | 감시: {top or '-'}")
wall = time.perf_counter() - t0
with open(a.csv, "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["file", "time_s", "sound", "score"])
    w.writerows(rows)
print(
    f"소리 {audio_s:.1f}초를 {wall:.2f}초에 처리(실시간의 {audio_s / wall:.0f}배),"
    f" NPU {np.mean(npu_ms):.2f} ms/조각 → {a.csv}"
)
