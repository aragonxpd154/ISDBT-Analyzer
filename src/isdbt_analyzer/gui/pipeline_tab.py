"""Aba "Pipeline ao vivo": a cadeia ISDB-T com constelações e contadores em tempo real."""

from __future__ import annotations

import math
import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..core import params as P
from ..core.analysis import LAYER_COLORS
from ..core.channel import carrier_frequencies
from ..core.fec import QEF_BER
from ..core.modulation import constellation, demap_hard, map_bits, mer_db, nearest_points, pam_table
from . import theme
from .fec_worker import FecWorker
from .state import AppState

BUFFER = 2400        # símbolos usados no cálculo do MER
PER_TICK = 300       # símbolos novos por quadro
BINS = 180           # resolução do mapa de densidade das constelações
DECAY = 0.82         # persistência (0 = sem memória, 1 = acumula para sempre)
BASE_INTERVAL = 80   # ms entre quadros (ajustado sozinho se a máquina for lenta)

STAGES = [
    ("ts", "TS\n188 B"),
    ("rs", "RS\n204,188"),
    ("conv", "Convol.\nK=7"),
    ("punc", "Puncion.\n{rate}"),
    ("intl", "Entrelaç.\nI={I}"),
    ("map", "Mapeam.\n{mod}"),
    ("ch", "Canal\n{ch}"),
    ("eq", "Equaliz."),
    ("demap", "Demap.\nLLR"),
    ("vit", "Viterbi"),
    ("rsdec", "RS\ndecod."),
    ("out", "TS\nsaída"),
]
STAGE_PLOT = {"map": 1, "ch": 2, "eq": 3, "demap": 4}


class PipelineDiagram(QWidget):
    """Diagrama de blocos com uma métrica ao vivo sob cada etapa."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(118)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.fmt = {"rate": "3/4", "I": 2, "mod": "64-QAM", "ch": "AWGN"}
        self.metrics: dict[str, tuple[str, str]] = {}

    def set_info(self, fmt: dict, metrics: dict) -> None:
        self.fmt = fmt
        self.metrics = metrics
        self.update()

    def paintEvent(self, _ev) -> None:  # noqa: N802
        c = theme.colors()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        n = len(STAGES)
        w = self.width()
        gap = 10
        bw = (w - gap * (n - 1) - 8) / n
        bh = 52
        y = 8
        tx_end = 6  # índice do canal: antes = transmissor, depois = receptor
        font = QFont(self.font())
        font.setPointSizeF(max(7.5, min(9.5, bw / 9)))
        p.setFont(font)
        for i, (key, label) in enumerate(STAGES):
            x = 4 + i * (bw + gap)
            rect = QRectF(x, y, bw, bh)
            if key == "ch":
                fill = QColor(theme.WARN)
                fill.setAlpha(60)
                border = QColor(theme.WARN)
            elif i < tx_end:
                fill = QColor(theme.ACCENT)
                fill.setAlpha(35)
                border = QColor(theme.ACCENT)
            else:
                fill = QColor("#8b7fd8")
                fill.setAlpha(35)
                border = QColor("#8b7fd8")
            path = QPainterPath()
            path.addRoundedRect(rect, 7, 7)
            p.fillPath(path, fill)
            p.setPen(QPen(border, 1.3))
            p.drawPath(path)
            p.setPen(QColor(c["text"]))
            try:
                text = label.format(**self.fmt)
            except (KeyError, IndexError):
                text = label
            p.drawText(rect, Qt.AlignCenter, text)
            if key in STAGE_PLOT:
                badge = QRectF(x + bw - 9, y - 7, 16, 16)
                p.setBrush(QColor(c["text"]))
                p.setPen(Qt.NoPen)
                p.drawEllipse(badge)
                p.setPen(QColor(c["bg"]))
                p.drawText(badge, Qt.AlignCenter, f"{STAGE_PLOT[key]}")
                p.setBrush(Qt.NoBrush)
            if i < n - 1:
                p.setPen(QPen(QColor(c["muted"]), 1.4))
                ax = x + bw + 1
                p.drawLine(QPointF(ax, y + bh / 2), QPointF(ax + gap - 3, y + bh / 2))
                p.drawLine(QPointF(ax + gap - 3, y + bh / 2), QPointF(ax + gap - 7, y + bh / 2 - 3))
                p.drawLine(QPointF(ax + gap - 3, y + bh / 2), QPointF(ax + gap - 7, y + bh / 2 + 3))
            m = self.metrics.get(key)
            if m:
                value, color = m
                p.setPen(QColor(color))
                f2 = QFont(font)
                f2.setBold(True)
                p.setFont(f2)
                p.drawText(QRectF(x - 6, y + bh + 6, bw + 12, 18), Qt.AlignCenter, value)
                p.setFont(font)
        p.setPen(QColor(c["muted"]))
        tx_w = tx_end * (bw + gap) - gap
        p.drawText(QRectF(4, y + bh + 26, tx_w, 16), Qt.AlignCenter, "TRANSMISSOR")
        p.drawText(QRectF(4 + (tx_end + 1) * (bw + gap), y + bh + 26, w - (tx_end + 1) * (bw + gap) - 8, 16),
                   Qt.AlignCenter, "RECEPTOR")
        p.end()


def _kpi(title: str) -> tuple[QFrame, QLabel, QLabel]:
    fr = QFrame()
    fr.setObjectName("kpi")
    v = QVBoxLayout(fr)
    v.setContentsMargins(10, 6, 10, 6)
    t = QLabel(title)
    t.setObjectName("kpiTitle")
    val = QLabel("—")
    val.setObjectName("kpiValue")
    val.setMinimumWidth(140)
    sub = QLabel("")
    sub.setObjectName("muted")
    v.addWidget(t)
    v.addWidget(val)
    v.addWidget(sub)
    return fr, val, sub


def _fmt_ber(x: float) -> str:
    if x is None or math.isnan(x):
        return "—"
    if x == 0:
        return "0"
    return f"{x:.1e}"


class ConstPlot(pg.PlotWidget):
    def __init__(self, title: str, lim: float = 1.6) -> None:
        super().__init__()
        self.setMinimumSize(200, 200)
        self.setAspectLocked(True)
        self.setMouseEnabled(False, False)
        self.hideButtons()
        self.setMenuEnabled(False)
        self.set_lim(lim)
        self.setTitle(title)
        # sem eixos numéricos: redesenhá-los a cada quadro era o maior custo da tela.
        # Uma cruz discreta em I=0 / Q=0 basta como referência.
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.axes_lines = [pg.InfiniteLine(pos=0, angle=ang, movable=False) for ang in (0, 90)]
        for ln in self.axes_lines:
            ln.setZValue(-2)
            self.addItem(ln)
        # pontos discretos (só a constelação ideal transmitida usa)
        self.scatter = pg.ScatterPlotItem(size=3, pen=None)
        self.addItem(self.scatter)
        # mapa de densidade com persistência: barato de desenhar e sem piscar
        self.img = pg.ImageItem()
        self.img.setZValue(-1)
        self.addItem(self.img)
        self.acc_ok = np.zeros((BINS, BINS), np.float32)
        self.acc_err = np.zeros((BINS, BINS), np.float32)
        self.ideal = pg.ScatterPlotItem(size=8, symbol="+", pen=pg.mkPen("#ffffff", width=1.2), brush=None)
        self.addItem(self.ideal)
        self.extra: list = []

    def set_lim(self, lim: float) -> None:
        """Mostra o quadrado [-lim, lim]² inteiro (o aspecto travado amplia o eixo que sobrar)."""
        changed = getattr(self, "lim", None) != lim
        self.lim = lim
        self.getViewBox().setRange(xRange=(-lim, lim), yRange=(-lim, lim), padding=0)
        if changed and hasattr(self, "img"):
            self.clear_density()

    def clear_density(self) -> None:
        self.acc_ok[:] = 0
        self.acc_err[:] = 0
        self.img.clear()

    def show_density(self, pts: np.ndarray, color: str, err: np.ndarray | None = None) -> None:
        """Acumula os pontos num histograma 2D com decaimento e desenha como imagem RGBA."""
        lim = self.lim
        rng = [[-lim, lim], [-lim, lim]]
        h, _, _ = np.histogram2d(pts.real, pts.imag, bins=BINS, range=rng)
        self.acc_ok *= DECAY
        self.acc_ok += h.astype(np.float32)
        self.acc_err *= DECAY
        if err is not None and err.size:
            he, _, _ = np.histogram2d(err.real, err.imag, bins=BINS, range=rng)
            self.acc_err += he.astype(np.float32)
        top = max(float(self.acc_ok.max()), 1.0)
        a = np.sqrt(np.clip(self.acc_ok / (0.35 * top), 0.0, 1.0))
        rgba = np.empty((BINS, BINS, 4), np.uint8)
        c = pg.mkColor(color)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = c.red(), c.green(), c.blue()
        rgba[..., 3] = (a * 255).astype(np.uint8)
        if err is not None:
            e = self.acc_err > 0.05
            # engrossa cada erro para 3×3 bins, para ficar bem visível
            d = e.copy()
            d[1:, :] |= e[:-1, :]
            d[:-1, :] |= e[1:, :]
            d[:, 1:] |= d[:, :-1].copy()
            d[:, :-1] |= d[:, 1:].copy()
            rgba[d] = (229, 72, 77, 255)
        self.img.setImage(rgba, autoLevels=False, levels=(0, 255))
        self.img.setRect(QRectF(-lim, -lim, 2 * lim, 2 * lim))

    def resizeEvent(self, ev) -> None:  # noqa: N802
        super().resizeEvent(ev)
        if hasattr(self, "lim"):
            self.set_lim(self.lim)

    def apply_theme(self) -> None:
        c = theme.colors()
        self.setBackground(c["plot_bg"])
        self.getPlotItem().titleLabel.setText(self.getPlotItem().titleLabel.text, color=c["text"], size="10pt")
        for ln in self.axes_lines:
            ln.setPen(pg.mkPen(c["grid"], width=1))
        self.ideal.setPen(pg.mkPen(c["text"], width=1.2))

    def clear_extra(self) -> None:
        for it in self.extra:
            self.removeItem(it)
        self.extra = []


class PipelineTab(QWidget):
    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.state = state
        self.rng = np.random.default_rng()
        self.worker: FecWorker | None = None
        self.paused = False
        self._reset_buffers()
        self._reset_counters()

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        head = QHBoxLayout()
        title = QLabel("<b>Cadeia ISDB-T em tempo real</b> — arraste o C/N no painel e veja cada etapa reagir")
        head.addWidget(title, 1)
        self.btn_pause = QPushButton("⏸ Pausar")
        self.btn_reset = QPushButton("↺ Zerar contadores")
        self.btn_shot = QPushButton("📷 Capturar")
        for b in (self.btn_pause, self.btn_reset, self.btn_shot):
            head.addWidget(b)
        root.addLayout(head)

        self.diagram = PipelineDiagram()
        root.addWidget(self.diagram)

        plots = QHBoxLayout()
        self.p_tx = ConstPlot("1 · Mapeado (transmitido)")
        self.p_rx = ConstPlot("2 · Após o canal (antes de equalizar)")
        self.p_eq = ConstPlot("3 · Equalizado")
        self.p_dec = ConstPlot("4 · Decisão (erros em vermelho)")
        for pw in (self.p_tx, self.p_rx, self.p_eq, self.p_dec):
            plots.addWidget(pw)
        root.addLayout(plots, 3)

        bottom = QHBoxLayout()
        self.p_h = pg.PlotWidget(title="Resposta do canal |H(f)|")
        self.p_h.setMouseEnabled(False, False)
        self.p_h.hideButtons()
        self.p_h.setLabel("bottom", "frequência (MHz)")
        self.p_h.setLabel("left", "dB")
        self.h_curve = self.p_h.plot(pen=pg.mkPen(theme.WARN, width=1.5))
        bottom.addWidget(self.p_h, 3)

        self.p_hist = pg.PlotWidget(title="Bytes errados por pacote na entrada do RS")
        self.p_hist.setMouseEnabled(False, False)
        self.p_hist.hideButtons()
        self.hist_bins = ["0", "1–2", "3–4", "5–6", "7–8", "9–16", "17–32", ">32"]
        self.p_hist.getAxis("bottom").setTicks([list(enumerate(self.hist_bins))])
        self.p_hist.setLabel("left", "pacotes (%)")
        self.hist_ok = pg.BarGraphItem(x=np.arange(5), height=np.zeros(5), width=0.75, brush=theme.GOOD)
        self.hist_bad = pg.BarGraphItem(x=np.arange(5, 8), height=np.zeros(3), width=0.75, brush=theme.BAD)
        self.p_hist.addItem(self.hist_ok)
        self.p_hist.addItem(self.hist_bad)
        self.thr_line = pg.InfiniteLine(pos=4.5, angle=90, pen=pg.mkPen(theme.WARN, width=1.5, style=Qt.DashLine),
                                        label="RS corrige até 8", labelOpts={"position": 0.88, "color": theme.WARN})
        self.p_hist.addItem(self.thr_line)
        self.p_hist.setYRange(0, 100, padding=0.02)
        bottom.addWidget(self.p_hist, 3)

        kpis = QGridLayout()
        self.k_mer = _kpi("MER (equalizado)")
        self.k_raw = _kpi("BER antes do Viterbi")
        self.k_vit = _kpi("BER após Viterbi")
        self.k_per = _kpi("PER após RS")
        self.k_pk = _kpi("Pacotes analisados")
        self.k_status = _kpi("Estado do serviço")
        for i, k in enumerate((self.k_mer, self.k_raw, self.k_vit, self.k_per, self.k_pk, self.k_status)):
            kpis.addWidget(k[0], i // 2, i % 2)
        kw = QWidget()
        kw.setLayout(kpis)
        bottom.addWidget(kw, 3)
        root.addLayout(bottom, 2)

        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_reset.clicked.connect(self.restart_worker_params)
        self.btn_shot.clicked.connect(self.snapshot)
        state.changed.connect(self.on_state)

        self.timer = QTimer(self)
        self.timer.setInterval(BASE_INTERVAL)
        self._cost = 0.0
        self._last_kpi = 0.0
        self._visual_key = None
        self.timer.timeout.connect(self.tick)
        self.apply_theme()

    # ---------------------------------------------------------------- ciclo de vida
    def start(self) -> None:
        if self.worker is None:
            self.worker = FecWorker()
        self.on_state()
        self.timer.start()

    def shutdown(self) -> None:
        self.timer.stop()
        if self.worker:
            self.worker.stop()
            self.worker = None

    def set_active(self, active: bool) -> None:
        """Só simula quando a aba está visível (economiza CPU)."""
        if active and not self.paused:
            self.timer.start()
            if self.worker:
                self.worker.pause(False)
        else:
            self.timer.stop()
            if self.worker:
                self.worker.pause(True)

    def apply_theme(self) -> None:
        c = theme.colors()
        for pw in (self.p_tx, self.p_rx, self.p_eq, self.p_dec):
            pw.apply_theme()
        for pw in (self.p_h, self.p_hist):
            pw.setBackground(c["plot_bg"])
            pw.getPlotItem().titleLabel.setText(pw.getPlotItem().titleLabel.text, color=c["text"], size="10pt")
            for ax in ("left", "bottom"):
                pw.getAxis(ax).setTextPen(c["muted"])
                pw.getAxis(ax).setPen(c["grid"])
        self.diagram.update()

    # ---------------------------------------------------------------- estado
    def _reset_buffers(self) -> None:
        self.buf_tx = np.zeros(0, complex)
        self.buf_eq = np.zeros(0, complex)
        self.local = {"bits": 0, "bit_errors": 0, "symbols": 0, "symbol_errors": 0}

    def _reset_counters(self) -> None:
        self.acc = {"bits": 0, "raw_bits": 0, "raw_bit_errors": 0, "vit_bit_errors": 0, "bytes_total": 0,
                    "vit_byte_errors": 0, "packets": 0, "rs_failures": 0,
                    "hist": np.zeros(P.RS_N + 1, dtype=np.int64)}

    def restart_worker_params(self) -> None:
        self._reset_counters()
        if self.worker:
            lay = self.state.focus()
            self.epoch = self.worker.set_params(self.state.cfg, self.state.profile(), lay.modulation,
                                                lay.code_rate, self.state.cn_db)

    def on_state(self) -> None:
        lay = self.state.focus()
        self._reset_buffers()
        self.restart_worker_params()
        self._last_kpi = 0.0
        for pw in (self.p_rx, self.p_eq, self.p_dec):
            pw.clear_density()
        st = self.state
        key = (lay, st.channel_name, st.sfn_delay_us, st.sfn_level_db, st.cfg.mode, st.cfg.guard,
               theme.is_dark())
        if key == self._visual_key:
            return  # só o C/N mudou: não precisa redesenhar eixos, rótulos e |H(f)|
        self._visual_key = key
        color = LAYER_COLORS.get(lay.name, theme.ACCENT)
        self.layer_color = color
        ideal, labels = constellation(lay.modulation)
        self.ideal = ideal
        self.p_tx.scatter.setData(ideal.real, ideal.imag, size=10, brush=pg.mkBrush(color), pen=None)
        self.p_tx.clear_extra()
        k = P.MODULATIONS[lay.modulation]
        if k <= 4:
            for pt, lab in zip(ideal, labels, strict=True):
                t = pg.TextItem(format(int(lab), f"0{k}b"), color=theme.colors()["muted"], anchor=(0.5, -0.4))
                t.setPos(pt.real, pt.imag)
                self.p_tx.addItem(t)
                self.p_tx.extra.append(t)
        for pw in (self.p_eq, self.p_dec):
            pw.ideal.setData(ideal.real, ideal.imag)
        # linhas de decisão
        self.p_dec.clear_extra()
        levels, _ = pam_table(k // 2)
        thr = (levels[:-1] + levels[1:]) / 2
        for t in thr:
            for ang in (0, 90):
                ln = pg.InfiniteLine(pos=t, angle=ang, pen=pg.mkPen(theme.colors()["muted"], width=0.8,
                                                                    style=Qt.DotLine))
                self.p_dec.addItem(ln)
                self.p_dec.extra.append(ln)
        # resposta do canal
        st = self.state.channel_state()
        self.p_rx.set_lim(1.6 if st.h.size == 1 else 2.6)
        f = carrier_frequencies(self.state.cfg) / 1e6
        if st.h.size == 1:
            self.h_curve.setData(f, np.zeros_like(f))
            self.p_h.setYRange(-20, 6)
        else:
            hdb = 20 * np.log10(np.maximum(np.abs(st.h), 1e-4))
            self.h_curve.setData(f, hdb)
            self.p_h.setYRange(max(-40, float(hdb.min()) - 3), float(hdb.max()) + 3)

    # ---------------------------------------------------------------- simulação
    def tick(self) -> None:
        t0 = time.perf_counter()
        lay = self.state.focus()
        k = P.MODULATIONS[lay.modulation]
        bits = self.rng.integers(0, 2, PER_TICK * k, dtype=np.uint8)
        sym = map_bits(bits, lay.modulation)
        rx, eq, _, _ = self.state.channel_state().apply(sym, self.state.cn_db, self.rng)
        dec_bits = demap_hard(eq, lay.modulation)
        bit_err = dec_bits != bits
        err = bit_err.reshape(-1, k).any(axis=1)
        self.local["bits"] += bits.size
        self.local["bit_errors"] += int(bit_err.sum())
        self.local["symbols"] += sym.size
        self.local["symbol_errors"] += int(err.sum())
        self.buf_tx = np.concatenate([self.buf_tx, sym])[-BUFFER:]
        self.buf_eq = np.concatenate([self.buf_eq, eq])[-BUFFER:]

        color = self.layer_color
        self.p_rx.show_density(rx, theme.WARN)
        self.p_eq.show_density(eq, color)
        self.p_dec.show_density(eq[~err], color, eq[err])

        # resultados do processo de FEC
        if self.worker:
            for epoch, r in self.worker.drain():
                if epoch != getattr(self, "epoch", -1):
                    continue
                for key in self.acc:
                    self.acc[key] = self.acc[key] + r[key]

        # números e diagrama: ~4 vezes por segundo bastam e evitam re-layout constante
        now = time.perf_counter()
        if now - self._last_kpi > 0.25:
            self._last_kpi = now
            self.update_kpis(lay)

        # se a máquina não acompanha, desacelera sozinho (a interface nunca trava)
        cost_ms = (time.perf_counter() - t0) * 1000
        self._cost = 0.8 * self._cost + 0.2 * cost_ms
        target = int(min(400, max(BASE_INTERVAL, self._cost * 3)))
        if abs(target - self.timer.interval()) > 15:
            self.timer.setInterval(target)

    def _set_label(self, label: QLabel, text: str, color: str | None = None) -> None:
        """Só mexe no widget quando algo muda (evita repaint e re-layout desnecessários)."""
        if label.text() != text:
            label.setText(text)
        if color is not None and label.property("_color") != color:
            label.setProperty("_color", color)
            label.setStyleSheet(f"color: {color}")

    def update_kpis(self, lay) -> None:
        a = self.acc
        lo = self.local
        muted = theme.colors()["muted"]
        if self.buf_eq.size:
            mer = mer_db(self.buf_eq, nearest_points(self.buf_eq, lay.modulation))
            mer_true = mer_db(self.buf_eq, self.buf_tx)
        else:
            mer = mer_true = float("nan")
        raw = lo["bit_errors"] / lo["bits"] if lo["bits"] else float("nan")
        sym_err = lo["symbol_errors"] / lo["symbols"] if lo["symbols"] else float("nan")
        vit = a["vit_bit_errors"] / a["bits"] if a["bits"] else float("nan")
        per = a["rs_failures"] / a["packets"] if a["packets"] else float("nan")

        self._set_label(self.k_mer[1], f"{mer_true:.1f} dB")
        self._set_label(self.k_mer[2], f"medido no receptor: {mer:.1f} dB")
        self._set_label(self.k_raw[1], _fmt_ber(raw))
        self._set_label(self.k_raw[2], f"símbolos errados: {sym_err * 100:.2f} %")
        vit_txt = _fmt_ber(vit)
        if a["bits"] and a["vit_bit_errors"] == 0:
            vit_txt = f"< {1 / a['bits']:.0e}"
        if math.isnan(vit):
            vit_color = muted
        else:
            vit_color = theme.GOOD if vit <= QEF_BER else theme.BAD
        self._set_label(self.k_vit[1], vit_txt, vit_color)
        self._set_label(self.k_vit[2], f"limite QEF: {QEF_BER:.0e}")
        if math.isnan(per):
            per_color = muted
        else:
            per_color = theme.GOOD if per == 0 else (theme.WARN if per < 0.01 else theme.BAD)
        self._set_label(self.k_per[1], "—" if math.isnan(per) else f"{per * 100:.3g} %", per_color)
        self._set_label(self.k_per[2], f"{a['rs_failures']} pacotes perdidos")
        self._set_label(self.k_pk[1], f"{a['packets']:,}".replace(",", "."))
        self._set_label(self.k_pk[2], f"{a['bits'] / 1e6:.1f} Mbit decodificados")

        if math.isnan(vit):
            status, color = "medindo…", muted
        elif per > 0.01:
            status, color = "SEM SERVIÇO", theme.BAD
        elif vit > QEF_BER or a["rs_failures"] > 0:
            status, color = "DEGRADADO", theme.WARN
        else:
            status, color = "QEF ✓", theme.GOOD
        self._set_label(self.k_status[1], status, color)
        self._set_label(self.k_status[2], f"C/N {self.state.cn_db:.1f} dB · {lay.modulation} {lay.code_rate}")

        # histograma
        h = a["hist"]
        tot = max(1, int(h.sum()))
        groups = [h[0], h[1:3].sum(), h[3:5].sum(), h[5:7].sum(), h[7:9].sum(), h[9:17].sum(), h[17:33].sum(),
                  h[33:].sum()]
        pct = np.array(groups, dtype=float) / tot * 100
        self.hist_ok.setOpts(height=pct[:5])
        self.hist_bad.setOpts(height=pct[5:])

        def health(v, good, warn):
            if v is None or math.isnan(v):
                return muted
            return theme.GOOD if v <= good else (theme.WARN if v <= warn else theme.BAD)

        st = self.state.channel_state()
        metrics = {
            "ts": (f"{self.state.cfg.layer_rate(lay) / 1e6:.2f} Mb/s", muted),
            "ch": (f"C/N {self.state.cn_db:.1f} dB", theme.WARN if st.isi_fraction == 0 else theme.BAD),
            "eq": (f"MER {mer_true:.1f}", theme.colors()["text"]),
            "demap": (_fmt_ber(raw), health(raw, 1e-3, 5e-2)),
            "vit": (vit_txt, health(vit, QEF_BER, 1e-2)),
            "rsdec": ("—" if math.isnan(per) else f"PER {per * 100:.2g}%", health(per, 0.0, 0.01)),
            "out": (status, color),
        }
        fmt = {"rate": lay.code_rate, "I": lay.interleave, "mod": lay.modulation,
               "ch": self.state.channel_name.replace("Brasil ", "Br. ")}
        if (fmt, metrics) != getattr(self, "_last_diagram", None):
            self._last_diagram = (fmt, metrics)
            self.diagram.set_info(fmt, metrics)

    # ---------------------------------------------------------------- ações
    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.btn_pause.setText("▶ Retomar" if self.paused else "⏸ Pausar")
        self.set_active(not self.paused)

    def snapshot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Salvar captura", "pipeline_isdbt.png", "PNG (*.png)")
        if path:
            self.grab().save(path)
