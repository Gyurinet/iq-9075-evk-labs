# CLIP 글 토크나이저(BPE): 영어 문장을 CLIP 텍스트 인코더의 입력(토큰 번호 77개)으로 바꾼다.
# OpenAI CLIP의 simple_tokenizer.py를 줄여 옮겼다(ftfy·regex 패키지 없이 동작).
#   원본: https://github.com/openai/CLIP  (MIT License, Copyright (c) 2021 OpenAI)
#   MIT License 전문은 data/CLIP_LICENSE에 있다.
import gzip
import html
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

VOCAB = Path(__file__).resolve().parents[1] / "data" / "bpe_simple_vocab_16e6.txt.gz"
SOT, EOT = 49406, 49407  # <|startoftext|>, <|endoftext|>


@lru_cache
def _byte_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs), strict=True))


class Tokenizer:
    PAT = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|[^\s\w]+", re.IGNORECASE)

    def __init__(self):
        merges = gzip.open(VOCAB).read().decode("utf-8").split("\n")[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(m.split()) for m in merges]
        vocab = list(_byte_unicode().values())
        vocab += [v + "</w>" for v in vocab] + ["".join(m) for m in merges]
        vocab += ["<|startoftext|>", "<|endoftext|>"]
        self.encoder = {v: i for i, v in enumerate(vocab)}
        self.ranks = {m: i for i, m in enumerate(merges)}
        self.cache = {}

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        while len(word) > 1:
            pairs = {(a, b) for a, b in zip(word, word[1:], strict=False)}
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    out.append(word[i] + word[i + 1])
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = tuple(out)
        self.cache[token] = word
        return word

    def encode(self, text):
        text = re.sub(r"\s+", " ", html.unescape(html.unescape(text))).strip().lower()
        ids = []
        for tok in self.PAT.findall(text):
            tok = "".join(_byte_unicode()[b] for b in tok.encode("utf-8"))
            ids += [self.encoder[t] for t in self._bpe(tok)]
        return ids

    def __call__(self, texts, length=77):
        """문장 목록 → [문장 수, 77] int32(앞에 시작, 뒤에 끝 토큰, 나머지는 0)"""
        out = np.zeros((len(texts), length), np.int32)
        for i, t in enumerate(texts):
            ids = [SOT, *self.encode(t)][: length - 1] + [EOT]
            out[i, : len(ids)] = ids
        return out
