"""Aba "Desempenho": curvas BER/PER, C/N mínimo por camada e exportação."""

from __future__ import annotations

import math

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.analysis import LAYER_COLORS, export_csv, fmt_prob, layer_performance, plot_performance
from . import theme
from .state import AppState


class PerformanceTab(QWidget):
    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.state = state
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        bar = QHBoxLayout()
        self.r_cn = QRadioButton("Eixo C/N")
        self.r_eb = QRadioButton("Eixo Eb/N0")
        self.r_cn.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.r_cn)
        grp.addButton(self.r_eb)
        self.x_min = QDoubleSpinBox()
        self.x_max = QDoubleSpinBox()
        for sp, v in ((self.x_min, 0.0), (self.x_max, 30.0)):
            sp.setRange(-10, 50)
            sp.setDecimals(1)
            sp.setSuffix(" dB")
            sp.setValue(v)
        self.btn_auto = QPushButton("Ajustar faixa")
        self.btn_png = QPushButton("Exportar gráfico…")
        self.btn_csv = QPushButton("Exportar CSV…")
        for w in (self.r_cn, self.r_eb, QLabel("de"), self.x_min, QLabel("até"), self.x_max, self.btn_auto):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.btn_png)
        bar.addWidget(self.btn_csv)
        root.addLayout(bar)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setStyleSheet("background: transparent")
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        root.addWidget(self.toolbar)
        root.addWidget(self.canvas, 3)

        self.note = QLabel()
        self.note.setObjectName("muted")
        self.note.setWordWrap(True)
        root.addWidget(self.note)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["Camada", "Configuração", "Taxa (Mb/s)", "C/N QEF", "Eb/N0 QEF", "Margem",
             "BER sem FEC", "BER Viterbi", "PER (RS)"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setMaximumHeight(130)
        root.addWidget(self.table)

        self.r_cn.toggled.connect(self.redraw)
        self.x_min.valueChanged.connect(self.redraw)
        self.x_max.valueChanged.connect(self.redraw)
        self.btn_auto.clicked.connect(self.auto_range)
        self.btn_png.clicked.connect(self.export_png)
        self.btn_csv.clicked.connect(self.export_csv)
        state.changed.connect(self.redraw)
        self._auto_done = False

    def auto_range(self) -> None:
        perf = layer_performance(self.state.cfg, self.state.table_channel() or "AWGN")
        q = [p.qef_esn0_db for p in perf if not math.isnan(p.qef_esn0_db)]
        if q:
            self.x_min.blockSignals(True)
            self.x_min.setValue(max(-5.0, math.floor(min(q) - 8)))
            self.x_min.blockSignals(False)
            self.x_max.setValue(math.ceil(max(max(q) + 6, self.state.cn_db + 2)))

    def redraw(self) -> None:
        if not self._auto_done:
            self._auto_done = True
            self.auto_range()
        st = self.state
        ch = st.table_channel()
        lo, hi = self.x_min.value(), self.x_max.value()
        if hi <= lo + 1:
            return
        plot_performance(self.fig, st.cfg, ch or "AWGN", lo, hi, "cn" if self.r_cn.isChecked() else "eb",
                         st.cn_db, dark=theme.is_dark())
        self.canvas.draw_idle()
        if ch is None:
            self.note.setText(f"⚠ O canal '{st.channel_name}' não tem curvas FEC calibradas; o gráfico mostra AWGN. "
                              "Use a aba Pipeline ao vivo para medir este canal.")
        else:
            self.note.setText("Linhas pontilhadas: sem FEC (fórmula exata). Contínuas: após Viterbi (Monte Carlo). "
                              "Tracejadas: após RS(204,188). Linha vermelha: C/N de operação. "
                              "QEF = BER 2×10⁻⁴ após o Viterbi.")
        self._fill_table(ch or "AWGN")

    def _fill_table(self, ch: str) -> None:
        perf = layer_performance(self.state.cfg, ch, self.state.cn_db)
        self.table.setRowCount(len(perf))
        for r, p in enumerate(perf):
            lay = p.layer
            m = round(p.margin(self.state.cn_db), 1) + 0.0
            vals = [
                lay.name, f"{lay.segments} seg · {lay.modulation} · {lay.code_rate} · I={lay.interleave}",
                f"{p.rate_bps / 1e6:.3f}", f"{p.qef_esn0_db:.1f} dB", f"{p.qef_ebn0_db:.1f} dB",
                f"{m:+.1f} dB", fmt_prob(p.raw_ber_at), fmt_prob(p.vit_ber_at),
                fmt_prob(p.per_at),
            ]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignCenter)
                if c == 0:
                    it.setForeground(QColor(LAYER_COLORS.get(lay.name, "#888")))
                if c == 5 and not math.isnan(m):
                    it.setForeground(QColor(theme.health_color(m)))
                self.table.setItem(r, c, it)

    def export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Exportar gráfico", "desempenho_isdbt.png",
                                              "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if not path:
            return
        fc = "#16181d" if theme.is_dark() else "white"
        self.fig.savefig(path, dpi=200, facecolor=fc, bbox_inches="tight")

    def export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Exportar CSV", "curvas_isdbt.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            export_csv(path, self.state.cfg, self.state.table_channel() or "AWGN",
                       self.x_min.value(), self.x_max.value())
        except OSError as exc:
            QMessageBox.critical(self, "Erro", str(exc))
