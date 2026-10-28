"""Painel lateral: modo, guarda, camadas A/B/C, canal, C/N e camada em foco."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core import params as P
from ..core.analysis import LAYER_COLORS, layer_performance
from ..core.channel import PROFILES
from ..core.ofdm import Layer, TransmissionConfig
from . import theme
from .state import SFN_NAME, AppState

DEFAULT_LAYERS = {
    "A": Layer("A", 1, "QPSK", "2/3", 4),
    "B": Layer("B", 12, "64-QAM", "3/4", 2),
    "C": Layer("C", 0, "16-QAM", "1/2", 2),
}


class LayerRow:
    def __init__(self, grid: QGridLayout, row: int, name: str, on_change) -> None:
        self.name = name
        d = DEFAULT_LAYERS[name]
        self.enable = QCheckBox(name)
        self.enable.setStyleSheet(f"QCheckBox {{ color: {LAYER_COLORS[name]}; font-weight: 700; }}")
        self.seg = QSpinBox()
        self.seg.setRange(1, 13)
        self.seg.setValue(max(1, d.segments))
        self.mod = QComboBox()
        self.mod.addItems(list(P.MODULATIONS))
        self.mod.setCurrentText(d.modulation)
        self.rate = QComboBox()
        self.rate.addItems(list(P.CODE_RATES))
        self.rate.setCurrentText(d.code_rate)
        self.inter = QComboBox()
        self.rate_lbl = QLabel("—")
        self.rate_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        for col, w in enumerate((self.enable, self.seg, self.mod, self.rate, self.inter, self.rate_lbl)):
            grid.addWidget(w, row, col)
        self.enable.setChecked(d.segments > 0)
        self.enable.toggled.connect(on_change)
        self.seg.valueChanged.connect(on_change)
        self.mod.currentTextChanged.connect(on_change)
        self.rate.currentTextChanged.connect(on_change)
        self.inter.currentTextChanged.connect(on_change)
        self._default_i = d.interleave

    def set_interleave_options(self, values, keep=True) -> None:
        cur = self.inter.currentText() if keep and self.inter.count() else str(self._default_i)
        self.inter.blockSignals(True)
        self.inter.clear()
        self.inter.addItems([str(v) for v in values])
        idx = self.inter.findText(cur)
        self.inter.setCurrentIndex(idx if idx >= 0 else min(2, len(values) - 1))
        self.inter.blockSignals(False)

    def set_enabled_fields(self, on: bool) -> None:
        for w in (self.seg, self.mod, self.rate, self.inter):
            w.setEnabled(on)

    def layer(self) -> Layer | None:
        if not self.enable.isChecked():
            return None
        return Layer(self.name, self.seg.value(), self.mod.currentText(), self.rate.currentText(),
                     int(self.inter.currentText() or 0))

    def load(self, layer: Layer | None) -> None:
        widgets = (self.enable, self.seg, self.mod, self.rate, self.inter)
        for w in widgets:
            w.blockSignals(True)
        self.enable.setChecked(layer is not None)
        if layer:
            self.seg.setValue(layer.segments)
            self.mod.setCurrentText(P.normalize_modulation(layer.modulation))
            self.rate.setCurrentText(P.normalize_code_rate(layer.code_rate))
            self.inter.setCurrentText(str(layer.interleave))
        for w in widgets:
            w.blockSignals(False)


class ConfigPanel(QWidget):
    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.state = state
        self.setMinimumWidth(380)
        self.setMaximumWidth(460)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 6, 0)

        # ---------------- sinal
        g = QGroupBox("Sinal OFDM")
        f = QFormLayout(g)
        self.mode = QComboBox()
        self.mode.addItems(["1 (2k)", "2 (4k)", "3 (8k)"])
        self.mode.setCurrentIndex(2)
        self.guard = QComboBox()
        self.guard.addItems(list(P.GUARD_INTERVALS))
        self.guard.setCurrentText("1/16")
        self.guard_info = QLabel()
        self.guard_info.setObjectName("muted")
        f.addRow("Modo", self.mode)
        f.addRow("Intervalo de guarda", self.guard)
        f.addRow("", self.guard_info)
        root.addWidget(g)

        # ---------------- camadas
        g = QGroupBox("Camadas hierárquicas")
        grid = QGridLayout(g)
        grid.setHorizontalSpacing(6)
        for col, t in enumerate(("Cam.", "Seg.", "Modulação", "Código", "I", "Mb/s")):
            lbl = QLabel(t)
            lbl.setObjectName("muted")
            grid.addWidget(lbl, 0, col)
        self.rows = [LayerRow(grid, i + 1, n, self._on_change) for i, n in enumerate("ABC")]
        self.total_lbl = QLabel()
        self.warn_lbl = QLabel()
        self.warn_lbl.setWordWrap(True)
        grid.addWidget(self.total_lbl, 4, 0, 1, 6)
        grid.addWidget(self.warn_lbl, 5, 0, 1, 6)
        root.addWidget(g)

        # ---------------- canal
        g = QGroupBox("Canal de propagação")
        f = QFormLayout(g)
        self.channel = QComboBox()
        self.channel.addItems(list(PROFILES) + [SFN_NAME])
        self.ch_desc = QLabel()
        self.ch_desc.setObjectName("muted")
        self.ch_desc.setWordWrap(True)
        sfn = QWidget()
        h = QHBoxLayout(sfn)
        h.setContentsMargins(0, 0, 0, 0)
        self.sfn_delay = QDoubleSpinBox()
        self.sfn_delay.setRange(0, 400)
        self.sfn_delay.setDecimals(1)
        self.sfn_delay.setSuffix(" µs")
        self.sfn_delay.setValue(state.sfn_delay_us)
        self.sfn_level = QDoubleSpinBox()
        self.sfn_level.setRange(-30, 0)
        self.sfn_level.setDecimals(1)
        self.sfn_level.setSuffix(" dB")
        self.sfn_level.setValue(state.sfn_level_db)
        h.addWidget(QLabel("atraso"))
        h.addWidget(self.sfn_delay)
        h.addWidget(QLabel("nível"))
        h.addWidget(self.sfn_level)
        self.sfn_box = sfn
        f.addRow("Perfil", self.channel)
        f.addRow(self.sfn_box)
        f.addRow(self.ch_desc)
        root.addWidget(g)

        # ---------------- operação
        g = QGroupBox("Ponto de operação")
        v = QVBoxLayout(g)
        h = QHBoxLayout()
        self.cn_slider = QSlider(Qt.Horizontal)
        self.cn_slider.setRange(0, 400)
        self.cn_slider.setValue(int(state.cn_db * 10))
        self.cn_spin = QDoubleSpinBox()
        self.cn_spin.setRange(0, 40)
        self.cn_spin.setDecimals(1)
        self.cn_spin.setSingleStep(0.1)
        self.cn_spin.setSuffix(" dB")
        self.cn_spin.setValue(state.cn_db)
        h.addWidget(QLabel("C/N"))
        h.addWidget(self.cn_slider, 1)
        h.addWidget(self.cn_spin)
        v.addLayout(h)
        h = QHBoxLayout()
        h.addWidget(QLabel("Camada em foco"))
        self.focus = QComboBox()
        h.addWidget(self.focus, 1)
        v.addLayout(h)
        self.margins = QFrame()
        self.margins.setObjectName("kpi")
        self.margins_layout = QVBoxLayout(self.margins)
        self.margins_layout.setContentsMargins(10, 8, 10, 8)
        v.addWidget(self.margins)
        root.addWidget(g)
        root.addStretch(1)

        # sinais
        self.mode.currentIndexChanged.connect(self._on_mode)
        self.guard.currentTextChanged.connect(self._on_change)
        self.channel.currentTextChanged.connect(self._on_channel)
        self.sfn_delay.valueChanged.connect(self._on_channel)
        self.sfn_level.valueChanged.connect(self._on_channel)
        self.cn_slider.valueChanged.connect(lambda v: self.cn_spin.setValue(v / 10))
        self.cn_spin.valueChanged.connect(self._on_cn)
        self.focus.currentIndexChanged.connect(self._on_focus)
        state.changed.connect(self.refresh_info)

        self._on_mode(keep=False)
        self._on_channel()
        self.refresh_info()

    # ---------------------------------------------------------------- eventos
    def _mode_number(self) -> int:
        return self.mode.currentIndex() + 1

    def _on_mode(self, *_args, keep=True) -> None:
        vals = P.MODES[self._mode_number()].interleave_lengths
        for r in self.rows:
            r.set_interleave_options(vals, keep=keep)
        self._on_change()

    def _on_channel(self, *_args) -> None:
        name = self.channel.currentText()
        self.sfn_box.setVisible(name == SFN_NAME)
        self.state.set_channel(name, self.sfn_delay.value(), self.sfn_level.value())

    def _on_focus(self, idx: int) -> None:
        if 0 <= idx < len(self.state.cfg.layers):
            self.state.set_focus(self.state.cfg.layers[idx].name)

    def _on_cn(self, value: float) -> None:
        self.cn_slider.blockSignals(True)
        self.cn_slider.setValue(int(round(value * 10)))
        self.cn_slider.blockSignals(False)
        self.state.set_cn(value)

    def _on_change(self, *_args) -> None:
        layers = []
        for r in self.rows:
            lay = r.layer()
            r.set_enabled_fields(lay is not None)
            if lay:
                layers.append(lay)
        if not layers:
            self.warn_lbl.setText(f"<span style='color:{theme.BAD}'>Ative pelo menos uma camada.</span>")
            return
        cfg = TransmissionConfig(self._mode_number(), self.guard.currentText(), tuple(layers))
        try:
            warns = cfg.validate()
        except ValueError as exc:
            self.warn_lbl.setText(f"<span style='color:{theme.BAD}'>⚠ {exc}</span>")
            return
        self.warn_lbl.setText("".join(f"<span style='color:{theme.WARN}'>⚠ {w}</span><br>" for w in warns))
        self.state.set_config(cfg)
        cur = self.state.focus_name
        self.focus.blockSignals(True)
        self.focus.clear()
        self.focus.addItems([lay.label for lay in cfg.layers])
        names = [lay.name for lay in cfg.layers]
        self.focus.setCurrentIndex(names.index(cur) if cur in names else len(names) - 1)
        self.focus.blockSignals(False)
        self.state.set_focus(names[self.focus.currentIndex()])

    # ---------------------------------------------------------------- exibição
    def load_state(self, cfg: TransmissionConfig, channel: str, cn: float | None) -> None:
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(cfg.mode - 1)
        self.mode.blockSignals(False)
        vals = P.MODES[cfg.mode].interleave_lengths
        by = {lay.name: lay for lay in cfg.layers}
        for r in self.rows:
            r.set_interleave_options(vals, keep=False)
            r.load(by.get(r.name))
        self.guard.setCurrentText(cfg.guard)
        idx = self.channel.findText(channel)
        if idx >= 0:
            self.channel.setCurrentIndex(idx)
        if cn is not None:
            self.cn_spin.setValue(cn)
        self._on_change()

    def refresh_info(self) -> None:
        st = self.state
        cfg = st.cfg
        self.guard_info.setText(
            f"Tg = {cfg.tg * 1e6:.1f} µs · SFN até {cfg.sfn_max_distance_m / 1e3:.1f} km · "
            f"Δf = {cfg.carrier_spacing:.1f} Hz")
        by = {lay.name: lay for lay in cfg.layers}
        for r in self.rows:
            lay = by.get(r.name)
            r.rate_lbl.setText(f"{cfg.layer_rate(lay) / 1e6:.3f}" if lay else "—")
        used = sum(lay.segments for lay in cfg.layers)
        self.total_lbl.setText(f"<b>{used}/13 segmentos · total {cfg.total_rate() / 1e6:.3f} Mb/s</b>")

        prof = st.profile()
        desc = prof.description
        if not prof.is_awgn:
            desc += f" · espalhamento {prof.max_delay_us():.1f} µs"
            if prof.max_delay_us() > cfg.tg * 1e6:
                desc += (f"<br><span style='color:{theme.BAD}'>⚠ Eco além da guarda "
                         f"({cfg.tg * 1e6:.1f} µs): interferência entre símbolos.</span>")
        if st.table_channel() is None:
            desc += (f"<br><span style='color:{theme.WARN}'>Curvas FEC calibradas só para AWGN e "
                     "Brasil A–E; neste canal o pipeline mede ao vivo.</span>")
        self.ch_desc.setText(desc)

        # margens por camada
        while self.margins_layout.count():
            w = self.margins_layout.takeAt(0).widget()
            if w:
                w.deleteLater()
        perf = layer_performance(cfg, st.table_channel() or "AWGN", st.cn_db) if st.table_channel() else []
        if not perf:
            lbl = QLabel("Margem: use o Pipeline ao vivo para este canal.")
            lbl.setObjectName("muted")
            self.margins_layout.addWidget(lbl)
        for p in perf:
            m = round(p.margin(st.cn_db), 1) + 0.0
            ok = not math.isnan(m)
            color = theme.health_color(m if ok else None)
            status = "QEF" if ok and m >= 0 else "abaixo do QEF"
            lbl = QLabel(
                f"<span style='color:{LAYER_COLORS[p.layer.name]}'>■</span> Camada {p.layer.name}: "
                f"precisa {p.qef_esn0_db:.1f} dB · margem "
                f"<b style='color:{color}'>{m:+.1f} dB</b> <span style='color:{color}'>({status})</span>")
            self.margins_layout.addWidget(lbl)
