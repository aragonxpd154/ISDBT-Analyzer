"""Estado compartilhado entre as abas (configuração, canal, C/N e camada em foco)."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import QObject, QTimer, Signal

from ..core.channel import PROFILES, ChannelProfile, ChannelState, sfn_echo
from ..core.ofdm import Layer, TransmissionConfig
from ..core.tables import Curve, get_curve, table_channels

SFN_NAME = "Eco SFN"


class AppState(QObject):
    #: emitido (com atraso curto, agrupando mudanças) sempre que algo muda
    changed = Signal()
    #: emitido imediatamente quando só o C/N muda (para animações)
    cn_changed = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.cfg = TransmissionConfig()
        self.channel_name = "AWGN"
        self.sfn_delay_us = 20.0
        self.sfn_level_db = -3.0
        self.cn_db = 20.0
        self.focus_name = "B"
        self._ch_cache: tuple | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self.changed.emit)

    # ------------------------------------------------------------ mudanças
    def notify(self) -> None:
        self._timer.start()

    def set_config(self, cfg: TransmissionConfig) -> None:
        self.cfg = cfg
        if self.focus_name not in [lay.name for lay in cfg.layers]:
            self.focus_name = cfg.layers[-1].name
        self.notify()

    def set_channel(self, name: str, delay_us: float | None = None, level_db: float | None = None) -> None:
        self.channel_name = name
        if delay_us is not None:
            self.sfn_delay_us = delay_us
        if level_db is not None:
            self.sfn_level_db = level_db
        self.notify()

    def set_cn(self, cn_db: float) -> None:
        self.cn_db = float(cn_db)
        self.cn_changed.emit(self.cn_db)
        self.notify()

    def set_focus(self, name: str) -> None:
        self.focus_name = name
        self.notify()

    # ------------------------------------------------------------ consultas
    def focus(self) -> Layer:
        for lay in self.cfg.layers:
            if lay.name == self.focus_name:
                return lay.normalized()
        return self.cfg.layers[-1].normalized()

    def profile(self) -> ChannelProfile:
        if self.channel_name == SFN_NAME:
            return sfn_echo(self.sfn_delay_us, self.sfn_level_db)
        return PROFILES[self.channel_name]

    def channel_state(self) -> ChannelState:
        key = (self.channel_name, self.sfn_delay_us, self.sfn_level_db, self.cfg.mode, self.cfg.guard)
        if not self._ch_cache or self._ch_cache[0] != key:
            self._ch_cache = (key, ChannelState.build(self.profile(), self.cfg))
        return self._ch_cache[1]

    def table_channel(self) -> str | None:
        """Nome do canal se houver curvas calibradas para ele."""
        return self.channel_name if self.channel_name in table_channels() else None

    def curve(self, layer: Layer | None = None) -> Curve | None:
        lay = layer or self.focus()
        ch = self.table_channel()
        if ch is None:
            return None
        try:
            return get_curve(lay.modulation, lay.code_rate, ch)
        except (KeyError, FileNotFoundError):
            return None

    def with_layer(self, name: str, **kw) -> TransmissionConfig:
        layers = tuple(replace(lay, **kw) if lay.name == name else lay for lay in self.cfg.layers)
        return replace(self.cfg, layers=layers)
