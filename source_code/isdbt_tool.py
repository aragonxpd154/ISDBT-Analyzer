"""
isdbt_tool.py
================

Módulo de previsão simplificada de desempenho ISDB-T usado pela GUI.
Aceita nomes de modulação flexíveis (QPSK, 16-QAM/16QAM/QAM16, 64-QAM/64QAM/QAM64),
normalizando tudo para chaves canônicas com hífen ASCII.

Funções principais usadas pela GUI:
- theoretical_ber(modulation, ebn0_db) -> np.ndarray
- degrade_image(image_L, modulation, ebn0_db) -> PIL.Image[L]
- compute_data_rate(modulation, coding_rate, guard_ratio) -> float

Constantes expostas:
- MODES, GUARD_INTERVALS, MODULATIONS, CODING_RATES
"""

from __future__ import annotations

import os
import datetime
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Iterable, Optional, List

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# Parâmetros de modo / guarda
# ============================================================

@dataclass(frozen=True)
class ModeParameters:
    mode: int
    subcarriers: int
    samples_per_symbol: int
    interleaver_depths: Tuple[int, ...]


MODES: Dict[int, ModeParameters] = {
    1: ModeParameters(mode=1, subcarriers=1405, samples_per_symbol=2048,
                      interleaver_depths=(0, 380, 760, 1520)),
    2: ModeParameters(mode=2, subcarriers=2809, samples_per_symbol=4096,
                      interleaver_depths=(0, 190, 380, 760)),
    3: ModeParameters(mode=3, subcarriers=5617, samples_per_symbol=8192,
                      interleaver_depths=(0, 95, 190, 380)),
}

GUARD_INTERVALS: Dict[str, float] = {
    "1/4":  1.0 / 4.0,
    "1/8":  1.0 / 8.0,
    "1/16": 1.0 / 16.0,
    "1/32": 1.0 / 32.0,
}

# ============================================================
# Modulações (usar SEMPRE hífen ASCII nas chaves)
# ============================================================

MODULATIONS: Dict[str, int] = {
    "QPSK":   2,   # bits por símbolo
    "16-QAM": 4,
    "64-QAM": 6,
}

CODING_RATES: Tuple[float, ...] = (1/2, 2/3, 3/4, 5/6, 7/8)

# ============================================================
# Normalizador de nomes de modulação
# ============================================================

_HYPHENS = {"–", "—", "-"}  # en/em/ASCII hyphens

def _canon_mod_name(s: str) -> str:
    """
    Canoniza 's' para comparação:
    - maiúsculas
    - normaliza hífens tipográficos para '-'
    - remove espaços
    - remove hífens na versão de comparação
    Ex.: '16-QAM'/'16-QAM'/'16 QAM' -> '16QAM'
    """
    if s is None:
        return ""
    s = str(s).upper()
    for h in _HYPHENS:
        s = s.replace(h, "-")
    s = s.replace(" ", "")
    return s.replace("-", "")

def normalize_modulation(name: str) -> str:
    """
    Converte vários apelidos (QAM16, 16QAM, 16-QAM, 16-QAM, 16 QAM...) em
    uma das chaves canônicas de MODULATIONS: 'QPSK', '16-QAM', '64-QAM'.
    """
    canon = _canon_mod_name(name)
    if canon == "QPSK":
        return "QPSK"
    if canon in ("16QAM", "QAM16"):
        return "16-QAM"
    if canon in ("64QAM", "QAM64"):
        return "64-QAM"
    # última tentativa: talvez já seja chave exata
    if isinstance(name, str) and name in MODULATIONS:
        return name
    raise ValueError(f"Unsupported modulation: {name}. Use QPSK, 16-QAM ou 64-QAM.")

# ============================================================
# Utilitários matemáticos
# ============================================================

def qfunc(x: np.ndarray) -> np.ndarray:
    """Q(x) = 0.5 * erfc(x/sqrt(2)) (vetorizado)."""
    x_arr = np.asarray(x, dtype=float)
    try:
        erfc_func = np.erfc  # type: ignore[attr-defined]
    except AttributeError:
        erfc_func = np.vectorize(math.erfc)
    return 0.5 * erfc_func(x_arr / math.sqrt(2.0))

# ============================================================
# BER teórica
# ============================================================

def theoretical_ber(modulation: str, ebn0_db: np.ndarray) -> np.ndarray:
    """
    BER aproximada para QPSK/16-QAM/64-QAM em AWGN.
    `modulation` pode ser 'QPSK', '16-QAM', '64-QAM' ou apelidos equivalentes.
    """
    modulation = normalize_modulation(modulation)
    k = MODULATIONS[modulation]     # bits por símbolo
    M = 2 ** k

    ebn0_linear = 10 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    esn0_linear = ebn0_linear * k   # Es/N0 = Eb/N0 * k

    if M == 4:  # QPSK
        x = np.sqrt(2 * esn0_linear)
        ber = qfunc(x)
    else:
        coeff = (2.0 / k) * (1.0 - 1.0 / np.sqrt(M))
        x = np.sqrt(3 * k / (M - 1) * esn0_linear)
        ber = coeff * qfunc(x)

    return np.clip(ber, 0, 1)

# ============================================================
# Constelação (útil para salvar figuras em simulações)
# ============================================================

def awgn_noise(s: np.ndarray, snr_db: float) -> np.ndarray:
    snr_linear = 10 ** (snr_db / 10.0)
    signal_power = np.mean(np.abs(s) ** 2)
    noise_power = signal_power / snr_linear
    noise_std = math.sqrt(noise_power / 2)
    noise = noise_std * (np.random.randn(*s.shape) + 1j * np.random.randn(*s.shape))
    return s + noise

def generate_constellation(modulation: str, ebn0_db: float, num_symbols: int = 1000) -> np.ndarray:
    modulation = normalize_modulation(modulation)
    k = MODULATIONS[modulation]
    M = 2 ** k

    symbols = np.random.randint(0, M, num_symbols)

    m_side = int(np.sqrt(M))
    i_bits = symbols % m_side
    q_bits = symbols // m_side
    levels = np.arange(m_side)
    amplitudes = 2 * levels + 1 - m_side
    symbols_i = amplitudes[i_bits]
    symbols_q = amplitudes[q_bits]
    const_points = symbols_i + 1j * symbols_q
    average_energy = (2 / 3) * (M - 1)
    const_points = const_points / math.sqrt(average_energy)

    ebn0_linear = 10 ** (ebn0_db / 10.0)
    esn0_linear = ebn0_linear * k
    noisy_const = awgn_noise(const_points, 10 * np.log10(esn0_linear))
    return noisy_const

# ============================================================
# Degradação de imagem (canal AWGN + detecção dura)
# ============================================================

def degrade_image(image: Image.Image, modulation: str, ebn0_db: float) -> Image.Image:
    """
    Transmite uma imagem em tons de cinza via canal AWGN simplificado, retornando a reconstrução.
    - image: PIL.Image em modo 'L'
    - modulation: aceita QPSK/16-QAM/64-QAM e apelidos
    - ebn0_db: Eb/N0 em dB
    """
    modulation = normalize_modulation(modulation)
    if image.mode != "L":
        raise ValueError("Image must be grayscale (mode 'L')")

    k = MODULATIONS[modulation]
    M = 2 ** k

    img_array = np.array(image, dtype=np.uint8)
    bits = np.unpackbits(img_array.flatten())

    remainder = bits.size % k
    if remainder != 0:
        bits = np.concatenate([bits, np.zeros(k - remainder, dtype=np.uint8)])

    bit_groups = bits.reshape(-1, k)
    binary_ints = (bit_groups.dot(1 << np.arange(k)[::-1])).astype(np.uint16)
    symbols = binary_ints ^ (binary_ints >> 1)   # Gray

    m_side = int(np.sqrt(M))
    i_bits = symbols % m_side
    q_bits = symbols // m_side
    levels = np.arange(m_side)
    amplitudes = 2 * levels + 1 - m_side
    symbols_i = amplitudes[i_bits]
    symbols_q = amplitudes[q_bits]
    const_points = symbols_i + 1j * symbols_q
    average_energy = (2 / 3) * (M - 1)
    const_points = const_points / math.sqrt(average_energy)

    ebn0_linear = 10 ** (ebn0_db / 10.0)
    esn0_linear = ebn0_linear * k
    noisy_symbols = awgn_noise(const_points, 10 * np.log10(esn0_linear))

    labels = np.arange(M, dtype=np.uint16)
    binary_labels = labels ^ (labels >> 1)  # Gray->binary mapeamento
    i_lab = labels % m_side
    q_lab = labels // m_side
    sym_i = amplitudes[i_lab]
    sym_q = amplitudes[q_lab]
    ideal_points = (sym_i + 1j * sym_q) / math.sqrt(average_energy)

    diffs = noisy_symbols[:, None] - ideal_points[None, :]
    closest = np.argmin(np.abs(diffs), axis=1).astype(np.uint16)

    gray_vals = binary_labels[closest]
    # Inversão de Gray (iterativa)
    binary_vals = gray_vals.copy()
    shift = 1
    while shift < k:
        binary_vals ^= (gray_vals >> shift)
        shift += 1

    recovered_bits = ((binary_vals[:, None] & (1 << np.arange(k)[::-1])) > 0).astype(np.uint8)
    recovered_bits = recovered_bits.reshape(-1)
    recovered_bits = recovered_bits[: img_array.size * 8]  # descarta padding

    recovered_bytes = np.packbits(recovered_bits)
    recovered_img_array = recovered_bytes.reshape(img_array.shape).astype(np.uint8)
    return Image.fromarray(recovered_img_array, mode="L")

# ============================================================
# Taxa de dados aproximada (modo 3, fórmula proporcional)
# ============================================================

def compute_data_rate(modulation: str, coding_rate: float, guard_ratio: float) -> float:
    """
    Estima a taxa bruta (Mb/s) para uma configuração (modo 3),
    reproduzindo os exemplos de referência.
    """
    modulation = normalize_modulation(modulation)
    if coding_rate <= 0 or coding_rate >= 1:
        # Ainda assim aceitamos 1/2, 2/3, 3/4, 5/6, 7/8
        pass

    base_rates = {
        "QPSK":   6.08,   # Mb/s (R=3/4, GI=1/8)
        "16-QAM": 12.17,
        "64-QAM": 18.25,
    }
    k = MODULATIONS[modulation]
    base_rate = base_rates[modulation]

    coding_scale = coding_rate / (3.0 / 4.0)
    guard_scale = (1 + 1/8) / (1 + guard_ratio)
    return base_rate * coding_scale * guard_scale

# ============================================================
# Função opcional para simular e salvar (útil em linha de comando)
# ============================================================

def simulate_and_save(
    mode: int,
    guard_label: str,
    modulation: str,
    coding_rate: float,
    interleaver_depth: int,
    ebn0_range: Iterable[float],
    image_path: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> str:
    """Executa uma simulação, salvando curva de BER, constelações e imagens."""
    modulation = normalize_modulation(modulation)

    if mode not in MODES:
        raise ValueError(f"Invalid mode: {mode}")
    if guard_label not in GUARD_INTERVALS:
        raise ValueError(f"Invalid guard interval: {guard_label}")
    if modulation not in MODULATIONS:
        raise ValueError(f"Unsupported modulation: {modulation}")
    if coding_rate not in CODING_RATES:
        raise ValueError(f"Unsupported coding rate: {coding_rate}")
    if interleaver_depth not in MODES[mode].interleaver_depths:
        raise ValueError(f"Interleaver depth {interleaver_depth} not allowed for mode {mode}")

    base_dir = results_dir if results_dir else "results"
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sim_name = f"mode{mode}_GI{guard_label}_mod{modulation}_R{coding_rate}_I{interleaver_depth}_{timestamp}"
    sim_dir = os.path.join(base_dir, sim_name)
    os.makedirs(sim_dir, exist_ok=True)

    ebn0_array = np.array(list(ebn0_range), dtype=float)

    ber_curve = theoretical_ber(modulation, ebn0_array)
    coding_gain_map = {1/2: 3.0, 2/3: 5.0, 3/4: 6.0, 5/6: 7.5, 7/8: 8.5}
    gain_db = coding_gain_map.get(coding_rate, 0.0)
    ber_curve_coded = theoretical_ber(modulation, ebn0_array + gain_db)

    plt.figure()
    plt.semilogy(ebn0_array, ber_curve, 'o-', label="Uncoded BER")
    plt.semilogy(ebn0_array, ber_curve_coded, 's--', label="Coded BER (approx.)")
    plt.title(f"BER vs Eb/N0 ({modulation}, R={coding_rate}, GI={guard_label})")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate")
    plt.grid(True, which="both")
    plt.legend()
    ber_path = os.path.join(sim_dir, "ber_curve.png")
    plt.savefig(ber_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Constelações em três pontos (início, meio, fim)
    for snr_db in ebn0_array[[0, len(ebn0_array) // 2, -1]]:
        const_points = generate_constellation(modulation, float(snr_db))
        plt.figure()
        plt.plot(const_points.real, const_points.imag, '.', markersize=2)
        plt.title(f"Constellation at Eb/N0 = {snr_db:.1f} dB ({modulation})")
        plt.xlabel("In-phase")
        plt.ylabel("Quadrature")
        plt.grid(True)
        const_path = os.path.join(sim_dir, f"constellation_{snr_db:.1f}dB.png")
        plt.savefig(const_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Imagem de referência
    if image_path is None:
        grad = np.linspace(0, 255, 256, dtype=np.uint8)
        img = Image.fromarray(np.tile(grad, (256, 1)), mode="L")
    else:
        img = Image.open(image_path).convert("L")

    for snr_db in [float(ebn0_array[0]), float(ebn0_array[-1])]:
        degraded = degrade_image(img, modulation, snr_db)
        w, h = img.size
        combo = Image.new("L", (w * 2 + 10, h))
        combo.paste(img, (0, 0))
        combo.paste(degraded, (w + 10, 0))
        out_path = os.path.join(sim_dir, f"image_comparison_{snr_db:.1f}dB.png")
        combo.save(out_path)

    data_rate = compute_data_rate(modulation, coding_rate, GUARD_INTERVALS[guard_label])

    report_path = os.path.join(sim_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"ISDB-T Simulation Report\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Subcarriers: {MODES[mode].subcarriers}\n")
        f.write(f"Guard Interval: {guard_label} (fraction={GUARD_INTERVALS[guard_label]:.4f})\n")
        f.write(f"Modulation: {modulation}\n")
        f.write(f"Bits per symbol: {MODULATIONS[modulation]}\n")
        f.write(f"Coding rate: {coding_rate}\n")
        f.write(f"Interleaver depth: {interleaver_depth}\n")
        f.write(f"Eb/N0 range: {list(map(float, ebn0_array))}\n")
        f.write(f"Estimated raw data rate: {data_rate:.2f} Mb/s\n")

    return sim_dir

# ============================================================
# CLI opcional
# ============================================================

def run_cli():
    print("ISDB-T Python Simulator/Analyser (simplificado)")
    print("Parâmetros padrão do exemplo. Resultados serão salvos em ./results\n")

    # Seleções simples para demonstração
    mode = 3
    guard_label = "1/8"
    modulation = "64-QAM"
    coding_rate = 3/4
    interleaver_depth = MODES[mode].interleaver_depths[0]
    ebn0_range = np.arange(0, 16, 2)

    out_dir = simulate_and_save(
        mode=mode,
        guard_label=guard_label,
        modulation=modulation,
        coding_rate=coding_rate,
        interleaver_depth=interleaver_depth,
        ebn0_range=ebn0_range,
        image_path=None,
    )
    print(f"Concluído. Resultados em: {out_dir}")

if __name__ == "__main__":
    run_cli()
