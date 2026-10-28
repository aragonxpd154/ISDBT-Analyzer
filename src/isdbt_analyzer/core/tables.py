"""Tabelas de desempenho pós-FEC (Monte Carlo) e interpolação.

As curvas de BER pós-Viterbi e de taxa de erro de byte são medidas uma vez
com :func:`calibrate` (comando ``isdbt-analyzer calibrar``) e gravadas em
``resources/fec_tables.json``. Em uso normal, o programa só interpola as
tabelas, o que é instantâneo.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

from . import params as P
from .channel import PROFILES, ChannelState
from .fec import QEF_BER, ChainResult, rs_output_ber, rs_packet_error_rate, simulate_chain
from .modulation import ber_uncoded
from .ofdm import TransmissionConfig

TABLE_FILE = "fec_tables.json"
FLOOR = 1e-12


@dataclass(frozen=True)
class Curve:
    esn0_db: np.ndarray
    vit_ber: np.ndarray
    byte_err: np.ndarray

    def _interp(self, values: np.ndarray, esn0_db) -> np.ndarray:
        x = np.atleast_1d(np.asarray(esn0_db, dtype=float))
        ly = np.log10(np.clip(values, FLOOR, 1.0))
        out = np.interp(x, self.esn0_db, ly)
        # extrapolação em queda (cascata) acima do último ponto medido
        if len(self.esn0_db) >= 2:
            slope = (ly[-1] - ly[-2]) / (self.esn0_db[-1] - self.esn0_db[-2])
            slope = min(slope, -0.5)
            hi = x > self.esn0_db[-1]
            out[hi] = ly[-1] + slope * (x[hi] - self.esn0_db[-1])
        return np.clip(10 ** out, FLOOR, 0.5)

    def ber(self, esn0_db) -> np.ndarray:
        return self._interp(self.vit_ber, esn0_db)

    def byte_error(self, esn0_db) -> np.ndarray:
        return self._interp(self.byte_err, esn0_db)

    def per(self, esn0_db) -> np.ndarray:
        return rs_packet_error_rate(self.byte_error(esn0_db))

    def rs_ber(self, esn0_db) -> np.ndarray:
        return rs_output_ber(self.byte_error(esn0_db), self.ber(esn0_db))

    def required_esn0(self, target_ber: float = QEF_BER) -> float:
        """Es/N0 (dB) em que a BER pós-Viterbi cruza o alvo (padrão: QEF 2e-4)."""
        ly = np.log10(np.clip(self.vit_ber, FLOOR, 1.0))
        t = np.log10(target_ber)
        for i in range(len(ly) - 1):
            if ly[i] >= t >= ly[i + 1] and ly[i] != ly[i + 1]:
                f = (ly[i] - t) / (ly[i] - ly[i + 1])
                return float(self.esn0_db[i] + f * (self.esn0_db[i + 1] - self.esn0_db[i]))
        return float("nan")


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def _key(channel: str, modulation: str, code_rate: str) -> str:
    return f"{channel}|{P.normalize_modulation(modulation)}|{P.normalize_code_rate(code_rate)}"


@lru_cache(maxsize=1)
def load_tables(path: str | None = None) -> dict:
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = resources.files("isdbt_analyzer.resources").joinpath(TABLE_FILE).read_text(encoding="utf-8")
    return json.loads(text)


def has_tables() -> bool:
    try:
        load_tables()
        return True
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False


def get_curve(modulation: str, code_rate: str, channel: str = "AWGN") -> Curve:
    data = load_tables()["curves"]
    key = _key(channel, modulation, code_rate)
    if key not in data:
        raise KeyError(f"Sem tabela para {key}. Rode 'isdbt-analyzer calibrar'.")
    c = data[key]
    return Curve(np.array(c["esn0_db"]), np.array(c["vit_ber"]), np.array(c["byte_err"]))


def table_channels() -> list[str]:
    try:
        return list(load_tables().get("channels", ["AWGN"]))
    except (FileNotFoundError, OSError):
        return ["AWGN"]


def uncoded_ber(modulation: str, esn0_db, channel: str = "AWGN", cfg: TransmissionConfig | None = None):
    prof = PROFILES[channel]
    if prof.is_awgn:
        return ber_uncoded(modulation, esn0_db)
    st = _channel_state_cached(channel, (cfg or TransmissionConfig()).mode, (cfg or TransmissionConfig()).guard)
    eff = np.array([st.effective_esn0_db(e) for e in np.atleast_1d(esn0_db)])
    g = st.power_gains()[:: max(1, st.h.size // 384)]
    return ber_uncoded(modulation, eff, g)


@lru_cache(maxsize=32)
def _channel_state_cached(channel: str, mode: int, guard: str) -> ChannelState:
    return ChannelState.build(PROFILES[channel], TransmissionConfig(mode=mode, guard=guard))


# ---------------------------------------------------------------------------
# Calibração por Monte Carlo
# ---------------------------------------------------------------------------

def _measure_curve(args) -> tuple[str, dict]:
    channel_name, modulation, code_rate, seed, max_packets, min_errors = args
    rng = np.random.default_rng(seed)
    st = ChannelState.build(PROFILES[channel_name], TransmissionConfig())
    # ponto de partida: onde a BER bruta ≈ 0,15
    grid = np.arange(-4.0, 40.0, 0.5)
    raw = uncoded_ber(modulation, grid, channel_name)
    start = float(grid[np.argmax(raw < 0.15)]) - 1.0
    es_list, ber_list, byte_list = [], [], []
    es = start
    zero_runs = 0
    while es < 45.0:
        acc = ChainResult()
        batch = 128
        while acc.packets < max_packets:
            r, _ = simulate_chain(modulation, code_rate, st, es, batch, rng)
            acc.add(r)
            if acc.vit_bit_errors >= min_errors:
                break
            batch = min(512, batch * 2)
        es_list.append(round(es, 2))
        ber_list.append(acc.vit_bit_errors / acc.bits)
        byte_list.append(acc.vit_byte_errors / acc.bytes_total)
        if acc.vit_bit_errors == 0:
            zero_runs += 1
            if zero_runs >= 1:
                # sem erros: registra limite superior e encerra
                ber_list[-1] = 0.5 / acc.bits
                byte_list[-1] = 0.5 / acc.bytes_total
                break
        if acc.vit_bit_errors / acc.bits < 3e-7:
            break
        es += 0.5
    return _key(channel_name, modulation, code_rate), {
        "esn0_db": es_list, "vit_ber": ber_list, "byte_err": byte_list}


def calibrate(output: str | Path, channels: list[str] | None = None, max_packets: int = 4000,
              min_errors: int = 400, workers: int | None = None, progress=print) -> Path:
    """Mede todas as curvas e grava o JSON."""
    channels = channels or list(PROFILES)
    jobs = []
    seed = 1000
    for ch in channels:
        for mod in P.MODULATIONS:
            for rate in P.CODE_RATES:
                seed += 1
                jobs.append((ch, mod, rate, seed, max_packets, min_errors))
    curves = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (key, curve) in enumerate(pool.map(_measure_curve, jobs), 1):
            curves[key] = curve
            progress(f"[{i}/{len(jobs)}] {key}  ({time.time() - t0:.0f}s)")
    data = {
        "version": 1,
        "description": ("BER pós-Viterbi (soft, max-log) e taxa de erro de byte por Monte Carlo; "
                        "C/N = Es/N0 nas portadoras de dados; entrelaçamento ideal."),
        "qef_ber": QEF_BER,
        "channels": channels,
        "curves": curves,
    }
    out = Path(output)
    out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    load_tables.cache_clear()
    return out
