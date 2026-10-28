"""Mapeamento QAM com código Gray por eixo (como no ISDB-T), demapeamento e BER exata.

Convenção de bits (ISDB-T): para k bits b0..b(k-1), os bits de índice par
vão para o eixo I e os ímpares para o eixo Q, cada eixo com Gray próprio.
As constelações são normalizadas para energia média unitária (Es = 1):
QPSK/√2, 16-QAM/√10, 64-QAM/√42.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from scipy.special import erfc

from . import params as P


def _q(x):
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


@cache
def pam_table(bits_per_axis: int):
    """Tabela de um eixo PAM com Gray.

    Retorna (levels, labels):
      levels[j]  amplitude normalizada do nível j (ordenado do menor ao maior)
      labels[j]  rótulo Gray (inteiro, MSB primeiro) do nível j
    """
    m = 1 << bits_per_axis
    raw = np.arange(m) * 2 - (m - 1)           # -m+1 ... m-1
    k = 2 * bits_per_axis
    M = 1 << k
    scale = np.sqrt(2.0 * (M - 1) / 3.0)       # energia média do QAM quadrado
    levels = raw / scale
    labels = np.arange(m) ^ (np.arange(m) >> 1)  # Gray
    return levels, labels


@cache
def _label_to_level(bits_per_axis: int):
    levels, labels = pam_table(bits_per_axis)
    lut = np.empty(len(levels))
    lut[labels] = levels
    return lut


def constellation(modulation: str) -> tuple[np.ndarray, np.ndarray]:
    """Pontos ideais (complexos) e seus rótulos de k bits."""
    k = P.MODULATIONS[P.normalize_modulation(modulation)]
    labels = np.arange(1 << k)
    bits = ((labels[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8)
    return map_bits(bits.reshape(-1), modulation), labels


def map_bits(bits: np.ndarray, modulation: str) -> np.ndarray:
    """bits (uint8, tamanho múltiplo de k) → símbolos complexos."""
    k = P.MODULATIONS[P.normalize_modulation(modulation)]
    h = k // 2
    b = np.asarray(bits, dtype=np.uint8).reshape(-1, k)
    weights = 1 << np.arange(h - 1, -1, -1)
    i_lab = b[:, 0::2] @ weights
    q_lab = b[:, 1::2] @ weights
    lut = _label_to_level(h)
    return lut[i_lab] + 1j * lut[q_lab]


def _slice_axis(x: np.ndarray, bits_per_axis: int) -> np.ndarray:
    """Decisão dura num eixo: devolve o rótulo Gray do nível mais próximo."""
    levels, labels = pam_table(bits_per_axis)
    m = len(levels)
    step = levels[1] - levels[0]
    idx = np.clip(np.rint((x - levels[0]) / step), 0, m - 1).astype(np.int64)
    return labels[idx]


def demap_hard(symbols: np.ndarray, modulation: str) -> np.ndarray:
    """Símbolos (equalizados) → bits por decisão dura."""
    k = P.MODULATIONS[P.normalize_modulation(modulation)]
    h = k // 2
    s = np.asarray(symbols)
    i_lab = _slice_axis(s.real, h)
    q_lab = _slice_axis(s.imag, h)
    out = np.empty((s.size, k), dtype=np.uint8)
    for j in range(h):
        shift = h - 1 - j
        out[:, 2 * j] = (i_lab >> shift) & 1
        out[:, 2 * j + 1] = (q_lab >> shift) & 1
    return out.reshape(-1)


def nearest_points(symbols: np.ndarray, modulation: str) -> np.ndarray:
    """Ponto ideal mais próximo de cada símbolo recebido."""
    k = P.MODULATIONS[P.normalize_modulation(modulation)]
    h = k // 2
    levels, _ = pam_table(h)
    step = levels[1] - levels[0]
    s = np.asarray(symbols)

    def snap(x):
        idx = np.clip(np.rint((x - levels[0]) / step), 0, len(levels) - 1).astype(np.int64)
        return levels[idx]
    return snap(s.real) + 1j * snap(s.imag)


def llr_maxlog(symbols: np.ndarray, modulation: str, noise_var) -> np.ndarray:
    """LLRs max-log por bit (positivo ⇒ bit 0 mais provável).

    symbols   símbolos já equalizados
    noise_var variância complexa do ruído após equalização (escalar ou por símbolo)
    """
    k = P.MODULATIONS[P.normalize_modulation(modulation)]
    h = k // 2
    levels, labels = pam_table(h)
    s = np.asarray(symbols)
    nv = np.broadcast_to(np.asarray(noise_var, dtype=float), s.shape).reshape(-1, 1)
    out = np.empty((s.size, k), dtype=np.float32)
    for axis, comp in enumerate((s.real, s.imag)):
        d2 = (comp.reshape(-1, 1) - levels.reshape(1, -1)) ** 2  # (N, m)
        for j in range(h):
            bit = (labels >> (h - 1 - j)) & 1
            d0 = d2[:, bit == 0].min(axis=1)
            d1 = d2[:, bit == 1].min(axis=1)
            out[:, 2 * j + axis] = ((d1 - d0) / nv[:, 0]).astype(np.float32)
    return out.reshape(-1)


def awgn(symbols: np.ndarray, esn0_db: float, rng: np.random.Generator) -> np.ndarray:
    """Adiciona ruído gaussiano complexo para Es/N0 dado (Es = 1)."""
    n0 = 10 ** (-esn0_db / 10)
    noise = rng.standard_normal(symbols.shape) + 1j * rng.standard_normal(symbols.shape)
    return symbols + np.sqrt(n0 / 2) * noise


# ---------------------------------------------------------------------------
# BER exata (sem codificação) em AWGN — soma exata sobre regiões de decisão
# ---------------------------------------------------------------------------

def _axis_bit_error_prob(bits_per_axis: int, sigma: np.ndarray) -> np.ndarray:
    """Probabilidade média de erro de bit num eixo PAM-Gray, para cada σ."""
    levels, labels = pam_table(bits_per_axis)
    m = len(levels)
    step = levels[1] - levels[0]
    thr = np.concatenate(([-np.inf], (levels[:-1] + step / 2), [np.inf]))
    sig = np.asarray(sigma, dtype=float).reshape(-1, 1)
    total = np.zeros(sig.shape[0])
    for j in range(m):  # nível transmitido
        lo = (thr[:-1].reshape(1, -1) - levels[j]) / sig
        hi = (thr[1:].reshape(1, -1) - levels[j]) / sig
        prob = _q(lo) - _q(hi)                      # (S, m) prob. de cair na região r
        ham = np.array([bin(labels[j] ^ labels[r]).count("1") for r in range(m)])
        total += prob @ ham
    return total / (m * bits_per_axis)


def ber_uncoded(modulation: str, esn0_db, channel_power_gains=None) -> np.ndarray:
    """BER exata do QAM-Gray sem FEC, em função de Es/N0 (C/N) em dB.

    Se `channel_power_gains` (|H|² por portadora) for dado, a BER é a média
    sobre as portadoras (canal seletivo em frequência, equalização ideal).
    """
    k = P.MODULATIONS[P.normalize_modulation(modulation)]
    es = 10 ** (np.atleast_1d(np.asarray(esn0_db, dtype=float)) / 10)
    if channel_power_gains is None:
        g = np.array([1.0])
    else:
        g = np.asarray(channel_power_gains, dtype=float)
    sigma = np.sqrt(1.0 / (2.0 * np.outer(es, g)))   # (S, G)
    p = _axis_bit_error_prob(k // 2, sigma.reshape(-1)).reshape(sigma.shape)
    return p.mean(axis=1)


def ebn0_to_esn0(ebn0_db, modulation: str, code_rate=None) -> np.ndarray:
    """Converte Eb/N0 (por bit de informação) em Es/N0.

    Sem `code_rate`: bits brutos (Es = k·Eb).
    Com `code_rate`: inclui código convolucional e RS 188/204.
    """
    k = P.MODULATIONS[P.normalize_modulation(modulation)]
    eff = k
    if code_rate is not None:
        eff = k * float(P.CODE_RATES[P.normalize_code_rate(code_rate)]) * P.RS_K / P.RS_N
    return np.asarray(ebn0_db, dtype=float) + 10 * np.log10(eff)


def esn0_to_ebn0(esn0_db, modulation: str, code_rate=None) -> np.ndarray:
    return np.asarray(esn0_db, dtype=float) - (ebn0_to_esn0(0.0, modulation, code_rate))


def mer_db(received: np.ndarray, reference: np.ndarray) -> float:
    """MER (dB) = potência do sinal ideal / potência do erro."""
    err = np.mean(np.abs(received - reference) ** 2)
    sig = np.mean(np.abs(reference) ** 2)
    if err <= 0:
        return float("inf")
    return float(10 * np.log10(sig / err))
