"""Aba "Parâmetros OFDM": tempos, taxas, mapa de segmentos e guarda × ecos (SFN)."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QHeaderView, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from ..core import params as P
from ..core.analysis import LAYER_COLORS, plot_segments
from . import theme
from .state import AppState


def _table(rows: int, cols: int, headers) -> QTableWidget:
    t = QTableWidget(rows, cols)
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    return t


class OfdmTab(QWidget):
    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.state = state
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        top = QHBoxLayout()
        self.t_sys = _table(0, 2, ["Parâmetro", "Valor"])
        self.t_sys.setMinimumWidth(360)
        top.addWidget(self.t_sys, 2)
        self.t_lay = _table(0, 7, ["Camada", "Seg.", "Modulação", "Código", "Taxa (Mb/s)", "TSP/quadro",
                                   "Atraso do entrelaçador"])
        self.t_lay.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        top.addWidget(self.t_lay, 3)
        root.addLayout(top, 2)
        self.fig = Figure(figsize=(9, 3.6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        root.addWidget(self.canvas, 3)
        state.changed.connect(self.request)
        self._dirty = True

    # ------------------------------------------------------------ redesenho só quando visível
    def request(self) -> None:
        if self.isVisible():
            self._dirty = False
            self.refresh()
        else:
            self._dirty = True

    def showEvent(self, ev) -> None:  # noqa: N802
        super().showEvent(ev)
        if getattr(self, "_dirty", True):
            self._dirty = False
            self.refresh()

    def refresh(self) -> None:
        st = self.state
        cfg = st.cfg
        s = cfg.summary()
        mp = cfg.mode_params
        rows = [
            ("Modo", f"{cfg.mode} · FFT {mp.fft_size} pontos"),
            ("Portadoras ativas", f"{mp.total_carriers} ({mp.carriers_per_seg}/segmento, "
                                  f"{mp.data_carriers_per_seg} de dados)"),
            ("Tempo útil Tu", f"{s['Tu_us']:.1f} µs"),
            ("Intervalo de guarda Tg", f"{cfg.guard} → {s['Tg_us']:.1f} µs"),
            ("Símbolo OFDM Ts", f"{s['Ts_us']:.1f} µs"),
            ("Espaçamento entre portadoras", f"{s['espacamento_Hz']:.3f} Hz"),
            ("Banda ocupada", f"{s['banda_ocupada_MHz']:.4f} MHz"),
            ("Quadro OFDM (204 símbolos)", f"{s['quadro_ms']:.2f} ms"),
            ("Diferença máx. de percurso (SFN)", f"{s['sfn_km']:.2f} km"),
            ("Taxa total", f"{s['taxa_total_Mbps']:.3f} Mb/s"),
        ]
        self.t_sys.setRowCount(len(rows))
        for r, (a, b) in enumerate(rows):
            self.t_sys.setItem(r, 0, QTableWidgetItem(a))
            it = QTableWidgetItem(b)
            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.t_sys.setItem(r, 1, it)
        self.t_lay.setRowCount(len(s["camadas"]))
        for r, c in enumerate(s["camadas"]):
            vals = [c["camada"], str(c["segmentos"]), c["modulacao"], c["codigo"], f"{c['taxa_Mbps']:.3f}",
                    f"{c['tsp_por_quadro']:.0f}",
                    f"I={c['I']} · {P.TIME_INTERLEAVER_DELAY_PER_I * c['I']} símb. · {c['atraso_entrelacador_ms']:.0f} ms"]
            for col, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignCenter)
                if col == 0:
                    it.setForeground(QColor(LAYER_COLORS.get(v, "#888")))
                self.t_lay.setItem(r, col, it)
        self._draw()

    def _draw(self) -> None:
        dark = theme.is_dark()
        fg = theme.colors()["text"]
        mut = theme.colors()["muted"]
        st = self.state
        cfg = st.cfg
        self.fig.clear()
        self.fig.patch.set_facecolor(theme.colors()["panel"])
        ax1 = self.fig.add_axes([0.02, 0.56, 0.96, 0.4])
        plot_segments(ax1, cfg, dark)

        ax = self.fig.add_axes([0.06, 0.12, 0.9, 0.3])
        ax.set_facecolor(theme.colors()["panel"])
        tg, tu = cfg.tg * 1e6, cfg.tu * 1e6
        prof = st.profile()
        d, a = prof.taps()
        delays = (d - d.min()) * 1e6
        levels = 20 * np.log10(np.abs(a) / np.abs(a).max() + 1e-12)
        ax.add_patch(Rectangle((0, 0.15), tg, 0.7, color=theme.WARN, alpha=0.45))
        ax.add_patch(Rectangle((tg, 0.15), tu, 0.7, color=theme.ACCENT, alpha=0.35))
        ax.text(tg / 2, 0.5, f"Guarda\n{tg:.1f} µs", ha="center", va="center", color=fg, fontsize=8)
        ax.text(tg + tu / 2, 0.5, f"Tempo útil Tu = {tu:.1f} µs", ha="center", va="center", color=fg, fontsize=9)
        xmax = max(tg * 1.6, delays.max() * 1.25 if delays.size else 0, tg + tu * 0.15)
        for dly, lv in zip(delays, levels, strict=True):
            col = theme.GOOD if dly <= tg else theme.BAD
            ax.vlines(dly, 0.0, 1.0, color=col, lw=2)
            ax.text(dly, 1.03, f"{dly:.1f} µs\n{lv:.0f} dB", ha="center", va="bottom", color=col, fontsize=7)
        ax.set_xlim(-xmax * 0.02, xmax)
        ax.set_ylim(0, 1.45)
        ax.set_yticks([])
        ax.set_xlabel("atraso (µs) — ecos do canal sobre o símbolo OFDM", color=mut)
        ax.tick_params(colors=mut)
        for sp in ax.spines.values():
            sp.set_color(theme.colors()["grid"])
        ax.set_title(f"Guarda × ecos — canal {prof.name}: verde = dentro da guarda, vermelho = gera ISI",
                     color=fg, fontsize=9)
        self.canvas.draw_idle()
