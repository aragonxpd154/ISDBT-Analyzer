"""Codificação de canal do ISDB-T: código convolucional K=7 (171,133 oct)
puncionado + Reed-Solomon (204,188, t=8).

Contém um codificador e um decodificador de Viterbi soft (max-log) vetorizados
em NumPy, que processam centenas de blocos em paralelo. Assim dá para medir a
BER pós-Viterbi por Monte Carlo em segundos.

Cada bloco do Viterbi carrega exatamente um pacote TS de 204 bytes (1632 bits)
mais 6 bits de cauda. Com isso o RS é avaliado direto: um pacote falha quando
tem mais de 8 bytes errados após o Viterbi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np

from . import params as P
from .channel import ChannelState
from .modulation import llr_maxlog, map_bits

K = 7
G1, G2 = 0o171, 0o133   # notação usual: o MSB do polinômio multiplica o bit de entrada atual


def _reverse_bits(value: int, width: int = K) -> int:
    return int(f"{value:0{width}b}"[::-1], 2)


# Os registradores abaixo guardam o bit mais recente no LSB, então usamos os
# polinômios espelhados. (A ordem importa: com puncionamento, a convenção
# errada gera outro código, com distância livre menor.)
_M1, _M2 = _reverse_bits(G1), _reverse_bits(G2)
N_STATES = 1 << (K - 1)
TAIL = K - 1
QEF_BER = 2e-4  # BER pós-Viterbi que leva a ~1e-11 após o RS (critério QEF)


def _parity(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    out = np.zeros_like(x)
    while np.any(x):
        out ^= x & 1
        x >>= 1
    return out


# Tabelas da treliça --------------------------------------------------------
_ns = np.arange(N_STATES)
_u = _ns & 1
_pred = np.stack([(_ns >> 1), (_ns >> 1) | (1 << (K - 2))])        # (2, 64) predecessores
_reg = (_pred << 1) | _u[None, :]                                  # registrador de 7 bits
_out_x = _parity(_reg & _M1)
_out_y = _parity(_reg & _M2)
_combo = (_out_x * 2 + _out_y).astype(np.int64)                    # (2, 64) índice 0..3


def encode(info_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Codifica blocos (B, L) já com cauda → (X, Y) cada um (B, L)."""
    b = np.asarray(info_bits, dtype=np.int64)
    B, L = b.shape
    regs = np.zeros((B, L), dtype=np.int64)
    padded = np.concatenate([np.zeros((B, K - 1), dtype=np.int64), b], axis=1)
    for i in range(K):  # registrador: bit mais recente no LSB
        regs |= padded[:, K - 1 - i: K - 1 - i + L] << i
    return _parity(regs & _M1).astype(np.uint8), _parity(regs & _M2).astype(np.uint8)


def puncture_mask(code_rate: str, n_info: int) -> np.ndarray:
    """Máscara booleana para a sequência intercalada [x0, y0, x1, y1, ...]."""
    px, py = P.PUNCTURE_PATTERNS[P.normalize_code_rate(code_rate)]
    period = len(px)
    reps = -(-n_info // period)
    mx = np.tile(np.array([c == "1" for c in px]), reps)[:n_info]
    my = np.tile(np.array([c == "1" for c in py]), reps)[:n_info]
    m = np.empty(2 * n_info, dtype=bool)
    m[0::2], m[1::2] = mx, my
    return m


def viterbi_decode(llr_x: np.ndarray, llr_y: np.ndarray) -> np.ndarray:
    """Viterbi soft max-log em lote. Entradas (B, L) com LLR>0 ⇒ bit 0.

    Supõe treliça terminada no estado 0 (cauda de 6 zeros). Devolve (B, L) bits.
    """
    B, L = llr_x.shape
    lx = llr_x.astype(np.float32)
    ly = llr_y.astype(np.float32)
    # métrica de ramo (correlação) para as 4 combinações (x,y) ∈ {00,01,10,11}
    bm = np.stack([lx + ly, lx - ly, -lx + ly, -lx - ly], axis=-1)   # (B, L, 4)
    pm = np.full((B, N_STATES), -1e9, dtype=np.float32)
    pm[:, 0] = 0.0
    dec = np.empty((L, B, N_STATES), dtype=bool)
    p0, p1 = _pred[0], _pred[1]
    c0, c1 = _combo[0], _combo[1]
    for t in range(L):
        b = bm[:, t, :]
        m0 = pm[:, p0] + b[:, c0]
        m1 = pm[:, p1] + b[:, c1]
        d = m1 > m0
        dec[t] = d
        pm = np.where(d, m1, m0)
        if t & 63 == 63:
            pm -= pm.max(axis=1, keepdims=True)
    # traceback a partir do estado 0
    out = np.empty((B, L), dtype=np.uint8)
    state = np.zeros(B, dtype=np.int64)
    rows = np.arange(B)
    for t in range(L - 1, -1, -1):
        out[:, t] = state & 1
        state = (state >> 1) | (dec[t, rows, state].astype(np.int64) << (K - 2))
    return out


# ---------------------------------------------------------------------------
# Simulação de ponta a ponta (bits → QAM → canal → LLR → Viterbi → RS)
# ---------------------------------------------------------------------------

@dataclass
class ChainResult:
    """Contadores de uma rodada da cadeia completa."""
    bits: int = 0
    raw_bit_errors: int = 0          # bits codificados antes do Viterbi (decisão dura)
    raw_bits: int = 0
    vit_bit_errors: int = 0          # bits de informação após o Viterbi
    bytes_total: int = 0
    vit_byte_errors: int = 0
    packets: int = 0
    rs_failures: int = 0             # pacotes com > 8 bytes errados
    byte_err_hist: np.ndarray = field(default_factory=lambda: np.zeros(P.RS_N + 1, dtype=np.int64))

    def add(self, other: ChainResult) -> None:
        for name in ("bits", "raw_bit_errors", "raw_bits", "vit_bit_errors", "bytes_total",
                     "vit_byte_errors", "packets", "rs_failures"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.byte_err_hist = self.byte_err_hist + other.byte_err_hist

    @property
    def raw_ber(self) -> float:
        return self.raw_bit_errors / self.raw_bits if self.raw_bits else float("nan")

    @property
    def vit_ber(self) -> float:
        return self.vit_bit_errors / self.bits if self.bits else float("nan")

    @property
    def byte_error_rate(self) -> float:
        return self.vit_byte_errors / self.bytes_total if self.bytes_total else float("nan")

    @property
    def per(self) -> float:
        return self.rs_failures / self.packets if self.packets else float("nan")


PACKET_BITS = P.RS_N * 8  # 1632 bits de um pacote RS(204,188)


def simulate_chain(modulation: str, code_rate: str, channel: ChannelState, esn0_db: float,
                   n_packets: int, rng: np.random.Generator,
                   keep_symbols: int = 0, byte_interleave: bool = True) -> tuple[ChainResult, dict]:
    """Roda `n_packets` pacotes de 204 bytes pela cadeia completa.

    Retorna (contadores, amostras) — `amostras` traz até `keep_symbols` símbolos
    de cada etapa para os diagramas de constelação.

    Com `byte_interleave`, os bytes errados são redistribuídos aleatoriamente
    entre os pacotes do lote antes do RS. Isso emula o entrelaçador de bytes do
    ISDB-T, que espalha as rajadas de erro do Viterbi por vários pacotes.
    """
    mod = P.normalize_modulation(modulation)
    rate = P.normalize_code_rate(code_rate)
    k = P.MODULATIONS[mod]
    L = PACKET_BITS + TAIL
    info = rng.integers(0, 2, (n_packets, L), dtype=np.uint8)
    info[:, PACKET_BITS:] = 0
    x, y = encode(info)
    coded = np.empty((n_packets, 2 * L), dtype=np.uint8)
    coded[:, 0::2], coded[:, 1::2] = x, y
    mask = puncture_mask(rate, L)
    tx_bits = coded[:, mask].reshape(-1)
    n_tx = tx_bits.size
    pad = (-n_tx) % k
    if pad:
        tx_bits = np.concatenate([tx_bits, rng.integers(0, 2, pad, dtype=np.uint8)])
    perm = rng.permutation(tx_bits.size)          # entrelaçamento de bits ideal
    inter = tx_bits[perm]
    sym = map_bits(inter, mod)
    rx, eq, nv, h = channel.apply(sym, esn0_db, rng)

    llr_int = llr_maxlog(eq, mod, nv)
    hard_int = (llr_int < 0).astype(np.uint8)
    llr = np.empty_like(llr_int)
    llr[perm] = llr_int
    hard = np.empty_like(hard_int)
    hard[perm] = hard_int

    res = ChainResult()
    res.raw_bits = n_tx
    res.raw_bit_errors = int(np.count_nonzero(hard[:n_tx] != tx_bits[:n_tx]))

    full = np.zeros((n_packets, 2 * L), dtype=np.float32)
    full[:, mask] = llr[:n_tx].reshape(n_packets, -1)
    dec = viterbi_decode(full[:, 0::2], full[:, 1::2])

    d = dec[:, :PACKET_BITS]
    s = info[:, :PACKET_BITS]
    err = d != s
    res.bits = d.size
    res.vit_bit_errors = int(err.sum())
    byte_err = err.reshape(n_packets, P.RS_N, 8).any(axis=2)
    if byte_interleave and n_packets > 1:
        byte_err = byte_err.reshape(-1)[rng.permutation(byte_err.size)].reshape(n_packets, P.RS_N)
    per_pkt = byte_err.sum(axis=1)
    res.bytes_total = byte_err.size
    res.vit_byte_errors = int(per_pkt.sum())
    res.packets = n_packets
    res.rs_failures = int(np.count_nonzero(per_pkt > P.RS_T))
    res.byte_err_hist = np.bincount(per_pkt, minlength=P.RS_N + 1)

    samples = {}
    if keep_symbols:
        n = min(keep_symbols, sym.size)
        samples = {"tx": sym[:n], "rx": rx[:n], "eq": eq[:n], "h": h[:n]}
    return res, samples


# ---------------------------------------------------------------------------
# Reed-Solomon com entrelaçamento de bytes ideal (bytes independentes)
# ---------------------------------------------------------------------------

def rs_packet_error_rate(byte_error_prob) -> np.ndarray:
    """P(pacote de 204 bytes com mais de 8 bytes errados)."""
    p = np.clip(np.atleast_1d(np.asarray(byte_error_prob, dtype=float)), 0.0, 1.0)
    j = np.arange(P.RS_T + 1)
    coeffs = np.array([comb(P.RS_N, int(i)) for i in j], dtype=float)
    ok = (coeffs[None, :] * p[:, None] ** j[None, :] * (1 - p[:, None]) ** (P.RS_N - j[None, :])).sum(axis=1)
    return np.clip(1.0 - ok, 0.0, 1.0)


def rs_output_ber(byte_error_prob, vit_ber) -> np.ndarray:
    """BER aproximada após o RS: só pacotes que falham mantêm os erros."""
    p = np.clip(np.atleast_1d(np.asarray(byte_error_prob, dtype=float)), 1e-300, 1.0)
    per = rs_packet_error_rate(p)
    # num pacote que falha, o número esperado de bytes errados é ≈ max(9, 204·p)
    bytes_bad = np.maximum(P.RS_T + 1, P.RS_N * p)
    bits_per_bad_byte = np.clip(np.asarray(vit_ber, dtype=float) / p * 8, 1.0, 8.0)
    return np.clip(per * bytes_bad * bits_per_bad_byte / (P.RS_N * 8), 0.0, 0.5)
