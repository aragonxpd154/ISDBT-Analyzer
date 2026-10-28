"""Relatório em PDF (várias páginas) gerado com Matplotlib — sem dependências extras."""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from .. import __version__
from . import params as P
from .analysis import LAYER_COLORS, layer_performance, plot_performance, plot_segments
from .channel import PROFILES, ChannelState
from .modulation import constellation, map_bits, mer_db
from .ofdm import TransmissionConfig


def _fmt(v, nd=2, unit=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{nd}f}{unit}"


def export_pdf(path: str | Path, cfg: TransmissionConfig, channel: str = "AWGN",
               cn_db: float | None = None, author: str = "") -> Path:
    path = Path(path)
    s = cfg.summary()
    perf = layer_performance(cfg, channel, cn_db)
    with PdfPages(path) as pdf:
        # ---------------- página 1: parâmetros
        fig = Figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, "ISDB-T Analyzer — Relatório de configuração", fontsize=16, weight="bold")
        fig.text(0.07, 0.925, f"Gerado em {dt.datetime.now():%d/%m/%Y %H:%M} · v{__version__}"
                 + (f" · {author}" if author else ""), fontsize=9, color="#555")
        rows = [
            ("Modo", f"{cfg.mode} ({cfg.mode_params.fft_size} pontos de FFT)"),
            ("Intervalo de guarda", f"{cfg.guard}  ({s['Tg_us']:.1f} µs)"),
            ("Tu / Ts", f"{s['Tu_us']:.1f} µs / {s['Ts_us']:.1f} µs"),
            ("Espaçamento entre portadoras", f"{s['espacamento_Hz']:.2f} Hz"),
            ("Banda ocupada", f"{s['banda_ocupada_MHz']:.3f} MHz"),
            ("Duração do quadro (204 símbolos)", f"{s['quadro_ms']:.1f} ms"),
            ("Distância máxima SFN (guarda)", f"{s['sfn_km']:.1f} km"),
            ("Canal analisado", f"{channel} — {PROFILES[channel].description}"),
            ("C/N de operação", _fmt(cn_db, 1, " dB")),
            ("Taxa total", f"{s['taxa_total_Mbps']:.3f} Mb/s"),
        ]
        ax = fig.add_axes([0.07, 0.6, 0.86, 0.3])
        ax.axis("off")
        t = ax.table(cellText=[[a, b] for a, b in rows], colWidths=[0.42, 0.58], loc="upper left",
                     cellLoc="left")
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        t.scale(1, 1.5)

        hdr = ["Camada", "Seg.", "Modulação", "Código", "I", "Taxa (Mb/s)", "C/N QEF", "Eb/N0 QEF",
               "Margem", "PER"]
        body = []
        for p in perf:
            lay = p.layer
            body.append([lay.name, lay.segments, lay.modulation, lay.code_rate, lay.interleave,
                         f"{p.rate_bps / 1e6:.3f}", _fmt(p.qef_esn0_db, 1, " dB"),
                         _fmt(p.qef_ebn0_db, 1, " dB"),
                         _fmt(p.margin(cn_db), 1, " dB") if cn_db is not None else "—",
                         f"{p.per_at * 100:.2g} %" if p.per_at is not None else "—"])
        ax2 = fig.add_axes([0.07, 0.42, 0.86, 0.15])
        ax2.axis("off")
        t2 = ax2.table(cellText=body, colLabels=hdr, loc="upper left", cellLoc="center")
        t2.auto_set_font_size(False)
        t2.set_fontsize(8)
        t2.scale(1, 1.5)
        for (r, c), cell in t2.get_celld().items():
            if r > 0 and c == 0:
                cell.set_facecolor(LAYER_COLORS.get(body[r - 1][0], "#ddd"))
        ax3 = fig.add_axes([0.07, 0.25, 0.86, 0.12])
        plot_segments(ax3, cfg)
        fig.text(0.07, 0.08,
                 "QEF: BER pós-Viterbi = 2×10⁻⁴ (≈ quase sem erros após o RS 204,188).\n"
                 "C/N = Es/N0 nas portadoras de dados; curvas FEC medidas por Monte Carlo\n"
                 "(Viterbi soft max-log, entrelaçamento ideal, estimação de canal perfeita).",
                 fontsize=8, color="#444")
        pdf.savefig(fig)

        # ---------------- página 2: curvas
        fig = Figure(figsize=(8.27, 11.69))
        qefs = [p.qef_esn0_db for p in perf if not math.isnan(p.qef_esn0_db)]
        lo = (min(qefs) - 8) if qefs else 0
        hi = (max(qefs) + 6) if qefs else 30
        plot_performance(fig, cfg, channel, lo, hi, "cn", cn_db)
        fig.patch.set_alpha(1)
        pdf.savefig(fig)

        # ---------------- página 3: constelações
        fig = Figure(figsize=(8.27, 4.2 * len(cfg.layers)))
        axes = fig.subplots(len(cfg.layers), 2, squeeze=False)
        rng = np.random.default_rng(7)
        st = ChannelState.build(PROFILES[channel], cfg)
        for row, p in zip(axes, perf, strict=True):
            lay = p.layer
            k = P.MODULATIONS[lay.modulation]
            ideal, _ = constellation(lay.modulation)
            cn = cn_db if cn_db is not None else (p.qef_esn0_db if not math.isnan(p.qef_esn0_db) else 20)
            sym = map_bits(rng.integers(0, 2, 3000 * k, dtype=np.uint8), lay.modulation)
            _, eq, _, _ = st.apply(sym, cn, rng)
            col = LAYER_COLORS.get(lay.name, "#555")
            for ax, data, title in ((row[0], ideal, "Transmitido"),
                                    (row[1], eq, f"Recebido a {cn:.1f} dB · MER {mer_db(eq, sym):.1f} dB")):
                ax.scatter(data.real, data.imag, s=3 if data.size > 100 else 30, color=col, alpha=0.6)
                ax.set_aspect("equal")
                ax.set_xlim(-1.6, 1.6)
                ax.set_ylim(-1.6, 1.6)
                ax.grid(alpha=0.3)
                ax.set_title(f"Camada {lay.name} {lay.modulation} — {title}", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
    return path
