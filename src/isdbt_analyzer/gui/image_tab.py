"""Aba "Imagem": transmite uma imagem pela cadeia e compara com a original."""

from __future__ import annotations

from PIL import Image
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core import media
from .state import AppState
from .tasks import Task, measure_point, start
from .widgets import ImageView, pil_to_qimage


def _work(img, luma, lay, chstate, cn, curve):
    raw, st_raw = media.degrade_uncoded(img, lay.modulation, cn, chstate, luma_only=luma)
    if curve is not None:
        per, be = float(curve.per(cn)[0]), float(curve.byte_error(cn)[0])
    else:
        per, be = measure_point(lay.modulation, lay.code_rate, chstate, cn)
    coded, st = media.degrade_coded(img, per, be, luma_only=luma)
    st["per_prevista"] = per
    return raw, st_raw, coded, st


class ImageTab(QWidget):
    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.state = state
        self.img = media.test_pattern()
        self.out = None
        self.token = 0
        self.busy = False
        self.pending = False

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        bar = QHBoxLayout()
        self.btn_open = QPushButton("Abrir imagem…")
        self.btn_pattern = QPushButton("Padrão de teste")
        self.chk_luma = QCheckBox("Degradar só luminância (preserva cor)")
        self.chk_luma.setChecked(False)
        self.btn_save = QPushButton("Salvar recebida (com FEC)…")
        for w in (self.btn_open, self.btn_pattern, self.chk_luma):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.btn_save)
        root.addLayout(bar)

        views = QHBoxLayout()
        self.v_orig = ImageView("Original")
        self.v_raw = ImageView("Recebida sem FEC")
        self.v_rx = ImageView("Recebida com FEC")
        views.addWidget(self.v_orig)
        views.addWidget(self.v_raw)
        views.addWidget(self.v_rx)
        root.addLayout(views, 1)

        self.info = QLabel()
        self.info.setWordWrap(True)
        root.addWidget(self.info)
        help_ = QLabel("No meio: erros de bit crus, sem correção (chuvisco). À direita: com FEC, o Viterbi e o RS limpam tudo "
                       "até o limite — abaixo dele os pacotes de 188 bytes se perdem em blocos: é o "
                       "“efeito penhasco” da TV digital.")
        help_.setObjectName("muted")
        help_.setWordWrap(True)
        root.addWidget(help_)

        self.btn_open.clicked.connect(self.open_image)
        self.btn_pattern.clicked.connect(self.use_pattern)
        self.btn_save.clicked.connect(self.save)
        self.chk_luma.toggled.connect(self.recompute)
        state.changed.connect(self.recompute)
        self.v_orig.set_image(pil_to_qimage(self.img))

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Abrir imagem", "",
                                              "Imagens (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if not path:
            return
        try:
            self.img = media.prepare(Image.open(path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Erro", f"Não foi possível abrir a imagem:\n{exc}")
            return
        self.v_orig.set_image(pil_to_qimage(self.img))
        self.v_orig.set_caption(f"Original · {self.img.width}×{self.img.height}")
        self.recompute()

    def use_pattern(self) -> None:
        self.img = media.test_pattern()
        self.v_orig.set_image(pil_to_qimage(self.img))
        self.v_orig.set_caption("Original · padrão de teste")
        self.recompute()

    def recompute(self) -> None:
        if not self.isVisible():
            self.pending = True
            return
        if self.busy:
            self.pending = True
            return
        self.busy = True
        self.pending = False
        self.token += 1
        st = self.state
        lay = st.focus()
        task = Task(self.token, _work, self.img, self.chk_luma.isChecked(), lay,
                    st.channel_state(), st.cn_db, st.curve(lay))
        task.signals.done.connect(self._done)
        task.signals.failed.connect(self._failed)
        self.info.setText("Processando…")
        start(task)

    def _done(self, _token, res) -> None:
        self.busy = False
        raw, s_raw, out, stt = res
        self.out = out
        lay = self.state.focus()
        self.v_raw.set_image(pil_to_qimage(raw))
        self.v_raw.set_caption(f"Sem FEC · BER {s_raw['ber']:.1e}")
        self.v_rx.set_image(pil_to_qimage(out))
        self.v_rx.set_caption(f"Com Viterbi + RS · PER {stt['per_prevista']:.1e}")
        txt = (f"C/N {self.state.cn_db:.1f} dB · camada {lay.name} ({lay.modulation} {lay.code_rate}) — "
               f"<b>sem FEC</b>: {s_raw['bytes_errados']:,} de {s_raw['bytes']:,} bytes errados · "
               f"<b>com FEC</b>: {stt['pacotes_perdidos']:,} de {stt['pacotes']:,} pacotes TS perdidos, "
               f"{stt['bytes_errados']:,} bytes errados").replace(",", ".")
        self.info.setText(txt)
        if self.pending:
            self.recompute()

    def _failed(self, _token, msg) -> None:
        self.busy = False
        self.info.setText(f"Erro: {msg}")

    def showEvent(self, ev) -> None:  # noqa: N802
        super().showEvent(ev)
        if self.pending or self.out is None:
            self.recompute()

    def save(self) -> None:
        if self.out is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Salvar imagem recebida", "recebida.png", "PNG (*.png)")
        if path:
            self.out.save(path)
