"""Análises de alto nível usadas pela interface, pela linha de comando e pelos relatórios."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

from . import params as P
from .fec import QEF_BER
from .modulation import ebn0_to_esn0, esn0_to_ebn0
from .ofdm import Layer, TransmissionConfig
from .tables import get_curve, has_tables, uncoded_ber

#: Ordem física dos segmentos no espectro (do menor para o maior em frequência).
SEGMENT_SPECTRUM_ORDER = (11, 9, 7, 5, 3, 1, 0, 2, 4, 6, 8, 10, 12)

LAYER_COLORS = {"A": "#f2a541", "B": "#4fb3bf", "C": "#b07fd8"}


def fmt_prob(x: float | None, nd: int = 1) -> str:
    """Formata BER/PER: valores no piso numérico viram '< 1e-12'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if x <= 1.01e-12:
        return "< 1e-12"
    return f"{x:.{nd}e}"


@dataclass
class LayerPerformance:
    layer: Layer
    rate_bps: float
    qef_esn0_db: float          # C/N para BER 2e-4 após Viterbi
    qef_ebn0_db: float
    per_at: float | None = None
    vit_ber_at: float | None = None
    raw_ber_at: float | None = None

    def margin(self, cn_db: float) -> float:
        return cn_db - self.qef_esn0_db


def layer_performance(cfg: TransmissionConfig, channel: str = "AWGN", cn_db: float | None = None):
    out = []
    for lay in cfg.layers:
        lay = lay.normalized()
        qef = float("nan")
        per = vit = None
        if has_tables():
            try:
                curve = get_curve(lay.modulation, lay.code_rate, channel)
                qef = curve.required_esn0(QEF_BER)
                if cn_db is not None:
                    per = float(curve.per(cn_db)[0])
                    vit = float(curve.ber(cn_db)[0])
            except KeyError:
                pass
        raw = float(uncoded_ber(lay.modulation, cn_db, channel, cfg)[0]) if cn_db is not None else None
        out.append(LayerPerformance(
            layer=lay, rate_bps=cfg.layer_rate(lay), qef_esn0_db=qef,
            qef_ebn0_db=float(esn0_to_ebn0(qef, lay.modulation, lay.code_rate)) if not math.isnan(qef) else qef,
            per_at=per, vit_ber_at=vit, raw_ber_at=raw))
    return out


def curves_for_layer(layer: Layer, esn0_db: np.ndarray, channel: str = "AWGN",
                     cfg: TransmissionConfig | None = None) -> dict:
    lay = layer.normalized()
    res = {"esn0_db": esn0_db,
           "ebn0_db": esn0_to_ebn0(esn0_db, lay.modulation, lay.code_rate),
           "raw_ber": uncoded_ber(lay.modulation, esn0_db, channel, cfg)}
    if has_tables():
        try:
            c = get_curve(lay.modulation, lay.code_rate, channel)
            res.update(vit_ber=c.ber(esn0_db), rs_ber=c.rs_ber(esn0_db), per=c.per(esn0_db),
                       qef=c.required_esn0())
        except KeyError:
            pass
    return res


def plot_performance(fig, cfg: TransmissionConfig, channel: str, x_min: float, x_max: float,
                     x_axis: str = "cn", current_cn: float | None = None, dark: bool = False):
    """Desenha BER (sem FEC, pós-Viterbi, pós-RS) e PER de todas as camadas."""
    fig.clear()
    fg = "#e6e6e6" if dark else "#222222"
    grid = "#555555" if dark else "#cccccc"
    ax = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax)
    es = np.arange(x_min, x_max + 1e-9, 0.1)
    for lay in cfg.layers:
        c = curves_for_layer(lay, es, channel, cfg)
        color = LAYER_COLORS.get(lay.name, "#888888")
        x = c["esn0_db"] if x_axis == "cn" else c["ebn0_db"]
        name = f"{lay.name} ({P.normalize_modulation(lay.modulation)} {P.normalize_code_rate(lay.code_rate)})"
        ax.semilogy(x, np.clip(c["raw_ber"], 1e-12, 1), color=color, lw=1.2, ls=":",
                    label=f"{name} sem FEC")
        if "vit_ber" in c:
            ax.semilogy(x, c["vit_ber"], color=color, lw=2.0, label=f"{name} pós-Viterbi")
            ax.semilogy(x, np.clip(c["rs_ber"], 1e-12, 1), color=color, lw=1.4, ls="--",
                        label=f"{name} pós-RS")
            ax2.plot(x, c["per"] * 100, color=color, lw=2.0, label=name)
            q = c["qef"]
            if not math.isnan(q):
                qx = q if x_axis == "cn" else float(esn0_to_ebn0(q, lay.modulation, lay.code_rate))
                for a in (ax, ax2):
                    a.axvline(qx, color=color, lw=0.9, ls="-.", alpha=0.8)
                ax.annotate(f"QEF {qx:.1f} dB", (qx, 2e-4), xytext=(4, 6), textcoords="offset points",
                            color=color, fontsize=8)
    ax.axhline(QEF_BER, color=fg, lw=0.6, alpha=0.5)
    if current_cn is not None and x_axis == "cn":
        for a in (ax, ax2):
            a.axvline(current_cn, color="#e5484d", lw=1.5)
    ax.set_ylim(1e-9, 0.5)
    ax.set_ylabel("BER")
    ax2.set_ylabel("PER após RS (%)")
    ax2.set_ylim(-3, 103)
    ax2.set_xlabel("C/N (dB)" if x_axis == "cn" else "Eb/N0 (dB, por bit de informação)")
    ax.set_title(f"Desempenho — canal {channel}", color=fg)
    for a in (ax, ax2):
        a.grid(True, which="both", color=grid, lw=0.5, alpha=0.7)
        a.tick_params(colors=fg)
        for s in a.spines.values():
            s.set_color(grid)
        a.yaxis.label.set_color(fg)
        a.xaxis.label.set_color(fg)
        a.set_facecolor("none")
    leg = ax.legend(fontsize=7, ncol=2, loc="lower left", framealpha=0.6)
    if dark:
        leg.get_frame().set_facecolor("#2a2d33")
        for t in leg.get_texts():
            t.set_color(fg)
    fig.patch.set_alpha(0)
    fig.tight_layout()
    return ax, ax2


def export_csv(path: str | Path, cfg: TransmissionConfig, channel: str, x_min: float, x_max: float,
               step: float = 0.25) -> Path:
    es = np.arange(x_min, x_max + 1e-9, step)
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["camada", "modulacao", "codigo", "canal", "C/N_dB", "Eb/N0_dB",
                    "BER_sem_FEC", "BER_pos_Viterbi", "BER_pos_RS", "PER"])
        for lay in cfg.layers:
            c = curves_for_layer(lay, es, channel, cfg)
            for i, e in enumerate(es):
                w.writerow([lay.name, P.normalize_modulation(lay.modulation),
                            P.normalize_code_rate(lay.code_rate), channel,
                            f"{e:.2f}", f"{c['ebn0_db'][i]:.2f}", f"{c['raw_ber'][i]:.3e}",
                            f"{c['vit_ber'][i]:.3e}" if "vit_ber" in c else "",
                            f"{c['rs_ber'][i]:.3e}" if "rs_ber" in c else "",
                            f"{c['per'][i]:.3e}" if "per" in c else ""])
    return path


def plot_segments(ax, cfg: TransmissionConfig, dark: bool = False):
    """Mapa dos 13 segmentos no espectro, coloridos por camada."""
    fg = "#e6e6e6" if dark else "#222222"
    owner = {}
    seg = 0
    for lay in cfg.layers:
        for _ in range(lay.segments):
            owner[seg] = lay.name
            seg += 1
    ax.clear()
    for pos, s in enumerate(SEGMENT_SPECTRUM_ORDER):
        name = owner.get(s)
        color = LAYER_COLORS.get(name, "#444444" if dark else "#dddddd")
        ax.add_patch(Rectangle((pos, 0), 0.92, 1, color=color, alpha=0.9 if name else 0.4))
        ax.text(pos + 0.46, 0.5, f"{s}\n{name or '—'}", ha="center", va="center", fontsize=8,
                color="#111111" if name else fg)
    ax.set_xlim(-0.2, 13.1)
    ax.set_ylim(0, 1.25)
    ax.axis("off")
    ax.set_title("Segmentos no espectro (centro = segmento 0)", color=fg, fontsize=9)


__all__ = ["fmt_prob", "layer_performance", "curves_for_layer", "plot_performance", "export_csv",
           "plot_segments", "ebn0_to_esn0", "esn0_to_ebn0", "LAYER_COLORS"]
