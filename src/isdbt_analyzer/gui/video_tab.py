"""Aba "Vídeo (TS)": codifica em MPEG-TS, corrompe pacotes pela PER e reproduz lado a lado."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..core import video
from .state import AppState
from .tasks import Task, measure_point, start
from .widgets import ImageView, bgr_to_qimage


def _run(src, rate, lay, chstate, cn, curve, workdir, progress):
    if curve is not None:
        per, be = float(curve.per(cn)[0]), float(curve.byte_error(cn)[0])
    else:
        progress(2, "Medindo PER neste canal (Monte Carlo)…")
        per, be = measure_point(lay.modulation, lay.code_rate, chstate, cn)
    res = video.simulate_video(src, rate, per, be, workdir=workdir, progress=progress)
    return res, per


class VideoTab(QWidget):
    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.state = state
        self.src: Path | None = None
        self.result = None
        self.caps = (None, None)
        self.workdir = Path(tempfile.mkdtemp(prefix="isdbt_video_"))
        self.n_frames = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        bar = QHBoxLayout()
        self.btn_open = QPushButton("Abrir vídeo…")
        self.btn_test = QPushButton("Usar vídeo de teste")
        self.src_lbl = QLabel("Fonte: vídeo de teste (barras + tom de 1 kHz)")
        self.src_lbl.setObjectName("muted")
        self.btn_sim = QPushButton("▶ Simular transmissão")
        self.btn_sim.setObjectName("primary")
        for w in (self.btn_open, self.btn_test, self.src_lbl):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.btn_sim)
        root.addLayout(bar)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        views = QHBoxLayout()
        self.v_orig = ImageView("TS original")
        self.v_rx = ImageView("TS recebido")
        views.addWidget(self.v_orig)
        views.addWidget(self.v_rx)
        root.addLayout(views, 1)

        ctl = QHBoxLayout()
        self.btn_play = QPushButton("⏸")
        self.btn_play.setFixedWidth(44)
        self.seek = QSlider(Qt.Horizontal)
        self.time_lbl = QLabel("0,0 s")
        self.btn_save = QPushButton("Salvar TS corrompido…")
        self.btn_folder = QPushButton("Abrir pasta")
        for w in (self.btn_play, self.seek, self.time_lbl, self.btn_save, self.btn_folder):
            ctl.addWidget(w, 1 if w is self.seek else 0)
        root.addLayout(ctl)

        self.info = QLabel("Escolha o C/N no painel e clique em “Simular transmissão”. O vídeo é codificado em "
                           "H.264/MPEG-TS na taxa da camada em foco; cada pacote de 188 bytes se perde com a "
                           "PER prevista e o resultado é decodificado de verdade (macroblocos, congelamentos).")
        self.info.setWordWrap(True)
        root.addWidget(self.info)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.btn_open.clicked.connect(self.open_video)
        self.btn_test.clicked.connect(self.use_test)
        self.btn_sim.clicked.connect(self.simulate)
        self.btn_play.clicked.connect(self.toggle_play)
        self.seek.sliderMoved.connect(self.seek_to)
        self.btn_save.clicked.connect(self.save_ts)
        self.btn_folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.workdir))))
        self._enable_player(False)
        if not video.find_ffmpeg():
            self.btn_sim.setEnabled(False)
            self.info.setText("⚠ FFmpeg não encontrado. Instale com:  sudo apt install ffmpeg")

    def _enable_player(self, on: bool) -> None:
        for w in (self.btn_play, self.seek, self.btn_save, self.btn_folder):
            w.setEnabled(on)

    def open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Abrir vídeo", "",
                                              "Vídeos (*.mp4 *.mkv *.mov *.avi *.webm *.ts *.m2ts *.mpg)")
        if path:
            self.src = Path(path)
            self.src_lbl.setText(f"Fonte: {self.src.name} (até 20 s)")

    def use_test(self) -> None:
        self.src = None
        self.src_lbl.setText("Fonte: vídeo de teste (barras + tom de 1 kHz)")

    def simulate(self) -> None:
        self.stop_player()
        st = self.state
        lay = st.focus()
        self.btn_sim.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.sim_params = (st.cn_db, lay)
        task = Task(1, _run, self.src, st.cfg.layer_rate(lay), lay, st.channel_state(), st.cn_db,
                    st.curve(lay), self.workdir, with_progress=True)
        task.signals.progress.connect(lambda p, m: (self.progress.setValue(p), self.progress.setFormat(f"{m} %p%")))
        task.signals.done.connect(self._done)
        task.signals.failed.connect(self._failed)
        start(task)

    def _failed(self, _t, msg: str) -> None:
        self.btn_sim.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Erro na simulação de vídeo", msg)

    def _done(self, _t, res) -> None:
        self.btn_sim.setEnabled(True)
        self.progress.setVisible(False)
        self.result, per = res
        r = self.result
        cn, lay = self.sim_params
        self.caps = (cv2.VideoCapture(str(r.original_view)), cv2.VideoCapture(str(r.corrupted_view)))
        self.n_frames = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for c in self.caps))
        self.seek.setRange(0, max(0, self.n_frames - 1))
        self.v_rx.set_caption(f"TS recebido · C/N {cn:.1f} dB · camada {lay.name} ({lay.modulation} {lay.code_rate})")
        s = r.stats
        self.info.setText(
            f"<b>PER prevista</b> {per:.2e} · <b>pacotes TS</b> {s.packets:,} · <b>corrompidos</b> {s.corrupted:,} "
            f"({s.per * 100:.3g} %) · vídeo H.264 a {r.video_bitrate / 1e6:.2f} Mb/s dentro de "
            f"{self.state.cfg.layer_rate(lay) / 1e6:.2f} Mb/s".replace(",", "."))
        self._enable_player(True)
        self.timer.start(int(1000 / r.fps))
        self.btn_play.setText("⏸")

    def next_frame(self) -> None:
        a, b = self.caps
        if a is None:
            return
        ok1, f1 = a.read()
        ok2, f2 = b.read()
        if not (ok1 and ok2):
            for c in self.caps:
                c.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        self.v_orig.set_image(bgr_to_qimage(f1))
        self.v_rx.set_image(bgr_to_qimage(f2))
        pos = int(a.get(cv2.CAP_PROP_POS_FRAMES))
        self.seek.blockSignals(True)
        self.seek.setValue(pos)
        self.seek.blockSignals(False)
        self.time_lbl.setText(f"{pos / self.result.fps:.1f} s".replace(".", ","))

    def toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("▶")
        elif self.result:
            self.timer.start(int(1000 / self.result.fps))
            self.btn_play.setText("⏸")

    def seek_to(self, pos: int) -> None:
        for c in self.caps:
            if c is not None:
                c.set(cv2.CAP_PROP_POS_FRAMES, pos)
        if not self.timer.isActive():
            self.next_frame()

    def stop_player(self) -> None:
        self.timer.stop()
        for c in self.caps:
            if c is not None:
                c.release()
        self.caps = (None, None)

    def save_ts(self) -> None:
        if not self.result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Salvar TS corrompido", "recebido.ts", "MPEG-TS (*.ts)")
        if path:
            shutil.copyfile(self.result.corrupted_ts, path)

    def hideEvent(self, ev) -> None:  # noqa: N802
        super().hideEvent(ev)
        if self.timer.isActive():
            self.toggle_play()

    def cleanup(self) -> None:
        self.stop_player()
        shutil.rmtree(self.workdir, ignore_errors=True)
