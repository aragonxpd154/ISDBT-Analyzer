"""Degradação de imagens pela cadeia ISDB-T.

Dois modos:

* **Sem FEC**: os bytes da imagem viram bits, passam por mapeamento QAM,
  canal e decisão dura. Os erros aparecem como "chuvisco" de bits.
* **Com FEC (Viterbi + RS)**: os bytes são divididos em pacotes TS de 188
  bytes. Cada pacote falha com a PER prevista para o C/N; os pacotes que não
  falham chegam perfeitos (efeito "penhasco" da TV digital).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from . import params as P
from .channel import ChannelState
from .modulation import demap_hard, map_bits

MAX_WORK_SIDE = 1280


def test_pattern(width: int = 960, height: int = 540) -> Image.Image:
    """Padrão de teste: barras coloridas, rampa de cinza e grade."""
    img = Image.new("RGB", (width, height), (16, 16, 16))
    d = ImageDraw.Draw(img)
    bars = [(191, 191, 191), (191, 191, 0), (0, 191, 191), (0, 191, 0),
            (191, 0, 191), (191, 0, 0), (0, 0, 191)]
    bw = width / len(bars)
    top_h = int(height * 0.62)
    for i, c in enumerate(bars):
        d.rectangle([int(i * bw), 0, int((i + 1) * bw), top_h], fill=c)
    ramp = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (int(height * 0.16), 1))
    img.paste(Image.fromarray(np.stack([ramp] * 3, axis=-1), "RGB"), (0, top_h))
    y0 = top_h + ramp.shape[0]
    for x in range(0, width, 24):
        d.line([x, y0, x, height], fill=(90, 90, 90))
    for y in range(y0, height, 24):
        d.line([0, y, width, y], fill=(90, 90, 90))
    d.ellipse([width // 2 - 60, y0 + 8, width // 2 + 60, height - 8], outline=(230, 230, 230), width=3)
    return img


def prepare(img: Image.Image) -> Image.Image:
    """Converte para RGB e limita o tamanho de trabalho (sem distorcer)."""
    img = img.convert("RGB")
    if max(img.size) > MAX_WORK_SIDE:
        img = img.copy()
        img.thumbnail((MAX_WORK_SIDE, MAX_WORK_SIDE), Image.Resampling.LANCZOS)
    return img


def _payload(img: Image.Image, luma_only: bool):
    if luma_only:
        ycc = img.convert("YCbCr")
        y, cb, cr = ycc.split()
        return np.asarray(y, dtype=np.uint8).copy(), (cb, cr)
    return np.asarray(img, dtype=np.uint8).copy(), None


def _rebuild(arr: np.ndarray, extra) -> Image.Image:
    if extra is not None:
        y = Image.fromarray(arr, "L")
        return Image.merge("YCbCr", (y, *extra)).convert("RGB")
    return Image.fromarray(arr, "RGB")


def degrade_uncoded(img: Image.Image, modulation: str, esn0_db: float, channel: ChannelState,
                    luma_only: bool = True, seed: int | None = None) -> tuple[Image.Image, dict]:
    """Transmite os bytes da imagem sem FEC (decisão dura)."""
    rng = np.random.default_rng(seed)
    mod = P.normalize_modulation(modulation)
    k = P.MODULATIONS[mod]
    arr, extra = _payload(img, luma_only)
    flat = arr.reshape(-1)
    bits = np.unpackbits(flat)
    pad = (-bits.size) % k
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    sym = map_bits(bits, mod)
    _, eq, _, _ = channel.apply(sym, esn0_db, rng)
    rx_bits = demap_hard(eq, mod)[: flat.size * 8]
    rx = np.packbits(rx_bits).reshape(arr.shape)
    bit_err = int(np.count_nonzero(rx_bits != bits[: flat.size * 8]))
    stats = {
        "bits": flat.size * 8,
        "bit_errors": bit_err,
        "ber": bit_err / (flat.size * 8),
        "bytes_errados": int(np.count_nonzero(rx != arr)),
        "bytes": flat.size,
    }
    return _rebuild(rx, extra), stats


def degrade_coded(img: Image.Image, per: float, byte_error: float, luma_only: bool = True,
                  seed: int | None = None) -> tuple[Image.Image, dict]:
    """Aplica perda de pacotes TS (188 bytes) conforme a PER prevista após o RS."""
    rng = np.random.default_rng(seed)
    arr, extra = _payload(img, luma_only)
    flat = arr.reshape(-1).copy()
    n_pk = -(-flat.size // P.TSP_BYTES)
    bad = rng.random(n_pk) < per
    # num pacote que falhou há mais de 8 bytes ruins em 204
    p_bad = max(byte_error, (P.RS_T + 1) / P.RS_N)
    idx_bad = np.flatnonzero(bad)
    for i in idx_bad:
        a, b = i * P.TSP_BYTES, min((i + 1) * P.TSP_BYTES, flat.size)
        hit = rng.random(b - a) < p_bad
        flat[a:b][hit] = rng.integers(0, 256, int(hit.sum()), dtype=np.uint8)
    rx = flat.reshape(arr.shape)
    stats = {
        "pacotes": int(n_pk),
        "pacotes_perdidos": int(bad.sum()),
        "per": float(bad.mean()) if n_pk else 0.0,
        "bytes_errados": int(np.count_nonzero(rx != arr)),
        "bytes": flat.size,
    }
    return _rebuild(rx, extra), stats


def side_by_side(a: Image.Image, b: Image.Image, gap: int = 8) -> Image.Image:
    w, h = a.size
    out = Image.new("RGB", (w * 2 + gap, h), (0, 0, 0))
    out.paste(a, (0, 0))
    out.paste(b, (w + gap, 0))
    return out
