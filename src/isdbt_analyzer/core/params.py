"""Parâmetros do sistema ISDB-T / ISDB-Tb (ARIB STD-B31 / ABNT NBR 15601).

Todos os valores aqui são definidos pela norma; nada é "ajustado" para bater
com exemplos. As taxas e tempos derivados ficam em :mod:`ofdm`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

#: Frequência de amostragem do IFFT (512/63 MHz), canal de 6 MHz.
SAMPLE_RATE_HZ = 512e6 / 63

#: Número total de segmentos OFDM num canal de 6 MHz.
N_SEGMENTS = 13

#: Símbolos OFDM por quadro.
SYMBOLS_PER_FRAME = 204

#: Bytes de um pacote TS e de um pacote TS com paridade RS(204,188).
TSP_BYTES = 188
RS_N, RS_K, RS_T = 204, 188, 8

#: Velocidade da luz (m/s) — para distância máxima de SFN.
C_LIGHT = 299_792_458.0


@dataclass(frozen=True)
class Mode:
    number: int
    fft_size: int                 # pontos da FFT
    data_carriers_per_seg: int    # portadoras de dados por segmento
    carriers_per_seg: int         # portadoras por segmento (dados + pilotos/controle)
    total_carriers: int           # portadoras ativas no canal (13 segmentos + 1 piloto contínuo)
    interleave_lengths: tuple[int, ...]  # valores permitidos de I (entrelaçamento temporal)


MODES: dict[int, Mode] = {
    1: Mode(1, 2048, 96, 108, 1405, (0, 4, 8, 16)),
    2: Mode(2, 4096, 192, 216, 2809, (0, 2, 4, 8)),
    3: Mode(3, 8192, 384, 432, 5617, (0, 1, 2, 4)),
}

#: Intervalos de guarda (fração do tempo útil Tu).
GUARD_INTERVALS: dict[str, Fraction] = {
    "1/4": Fraction(1, 4),
    "1/8": Fraction(1, 8),
    "1/16": Fraction(1, 16),
    "1/32": Fraction(1, 32),
}

#: Modulações de portadora e bits por símbolo.
MODULATIONS: dict[str, int] = {"QPSK": 2, "16-QAM": 4, "64-QAM": 6}

#: Taxas do código convolucional interno (K=7, 171/133 octal) e
#: padrões de puncionamento (X / Y) da norma.
CODE_RATES: dict[str, Fraction] = {
    "1/2": Fraction(1, 2),
    "2/3": Fraction(2, 3),
    "3/4": Fraction(3, 4),
    "5/6": Fraction(5, 6),
    "7/8": Fraction(7, 8),
}

PUNCTURE_PATTERNS: dict[str, tuple[str, str]] = {
    "1/2": ("1", "1"),
    "2/3": ("10", "11"),
    "3/4": ("101", "110"),
    "5/6": ("10101", "11010"),
    "7/8": ("1000101", "1111010"),
}

#: Atraso do entrelaçador temporal em símbolos OFDM = 95 × I (qualquer modo).
TIME_INTERLEAVER_DELAY_PER_I = 95


def normalize_modulation(name: str) -> str:
    """Aceita 'qpsk', '16qam', 'QAM16', '64 QAM', '16–QAM'... e devolve a chave canônica."""
    s = str(name).upper().replace(" ", "")
    for h in ("–", "—", "-", "_"):
        s = s.replace(h, "")
    if s == "QPSK":
        return "QPSK"
    if s in ("16QAM", "QAM16"):
        return "16-QAM"
    if s in ("64QAM", "QAM64"):
        return "64-QAM"
    raise ValueError(f"Modulação não suportada: {name!r}. Use QPSK, 16-QAM ou 64-QAM.")


def normalize_code_rate(rate) -> str:
    """Aceita '3/4', 0.75, Fraction(3, 4) e devolve a chave canônica '3/4'."""
    if isinstance(rate, str) and rate.strip() in CODE_RATES:
        return rate.strip()
    try:
        fr = Fraction(rate).limit_denominator(8)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Taxa de código inválida: {rate!r}") from exc
    for key, val in CODE_RATES.items():
        if val == fr:
            return key
    raise ValueError(f"Taxa de código não suportada: {rate!r}. Use 1/2, 2/3, 3/4, 5/6 ou 7/8.")


def normalize_guard(gi) -> str:
    if isinstance(gi, str) and gi.strip() in GUARD_INTERVALS:
        return gi.strip()
    fr = Fraction(gi).limit_denominator(32)
    for key, val in GUARD_INTERVALS.items():
        if val == fr:
            return key
    raise ValueError(f"Intervalo de guarda inválido: {gi!r}. Use 1/4, 1/8, 1/16 ou 1/32.")
