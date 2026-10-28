"""Janela principal do ISDB-T Analyzer."""

from __future__ import annotations

import multiprocessing
import os
import sys
from importlib import resources

from PySide6.QtCore import QSettings, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QTabWidget,
    QWidget,
)

from .. import __app_name__, __author__, __url__, __version__
from ..core import config_io
from ..core.tables import has_tables
from . import theme
from .config_panel import ConfigPanel
from .image_tab import ImageTab
from .ofdm_tab import OfdmTab
from .perf_tab import PerformanceTab
from .pipeline_tab import PipelineTab
from .state import AppState
from .video_tab import VideoTab


def _icon() -> QIcon:
    try:
        path = resources.files("isdbt_analyzer.resources").joinpath("icon.png")
        return QIcon(str(path))
    except Exception:  # noqa: BLE001
        return QIcon()


class MainWindow(QMainWindow):
    def __init__(self, dark: bool = True) -> None:
        super().__init__()
        self.setWindowTitle(f"{__app_name__} {__version__}")
        self.setWindowIcon(_icon())
        self.settings = QSettings("ObelLab", "ISDBT-Analyzer")
        self.state = AppState()

        central = QWidget()
        central.setObjectName("root")
        lay = QHBoxLayout(central)
        lay.setContentsMargins(10, 10, 10, 6)
        self.panel = ConfigPanel(self.state)
        scroll = QScrollArea()
        scroll.setWidget(self.panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(400)
        scroll.setMaximumWidth(480)
        lay.addWidget(scroll)

        self.tabs = QTabWidget()
        self.pipeline = PipelineTab(self.state)
        self.perf = PerformanceTab(self.state)
        self.image = ImageTab(self.state)
        self.video = VideoTab(self.state)
        self.ofdm = OfdmTab(self.state)
        self.tabs.addTab(self.pipeline, "Pipeline ao vivo")
        self.tabs.addTab(self.perf, "Desempenho BER/PER")
        self.tabs.addTab(self.image, "Imagem")
        self.tabs.addTab(self.video, "Vídeo (TS)")
        self.tabs.addTab(self.ofdm, "Parâmetros OFDM")
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.tabs.currentChanged.connect(self._tab_changed)

        self._build_menu()
        self.statusBar().showMessage(f"{__app_name__} v{__version__} · {__author__} · GPL-3.0")
        if not has_tables():
            QMessageBox.warning(self, "Tabelas ausentes",
                                "As tabelas de FEC não foram encontradas. Rode 'isdbt-analyzer calibrar'.")
        geo = self.settings.value("geometry")
        if geo:
            self.restoreGeometry(geo)
        else:
            self.resize(1500, 920)
        self.pipeline.start()
        self.state.changed.emit()

    # ---------------------------------------------------------------- menus
    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("&Arquivo")
        for text, slot, key in (
            ("Abrir configuração…", self.open_config, QKeySequence.Open),
            ("Salvar configuração…", self.save_config, QKeySequence.Save),
            ("Exportar relatório PDF…", self.export_report, "Ctrl+E"),
        ):
            a = QAction(text, self)
            a.setShortcut(key)
            a.triggered.connect(slot)
            m.addAction(a)
        m.addSeparator()
        a = QAction("Sair", self)
        a.setShortcut(QKeySequence.Quit)
        a.triggered.connect(self.close)
        m.addAction(a)

        m = self.menuBar().addMenu("&Exibir")
        self.act_dark = QAction("Tema escuro", self, checkable=True)
        self.act_dark.setChecked(theme.is_dark())
        self.act_dark.toggled.connect(self.set_dark)
        m.addAction(self.act_dark)

        m = self.menuBar().addMenu("A&juda")
        a = QAction("Repositório no GitHub", self)
        a.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(__url__)))
        m.addAction(a)
        a = QAction("Sobre…", self)
        a.triggered.connect(self.about)
        m.addAction(a)

    def set_dark(self, dark: bool) -> None:
        theme.apply(QApplication.instance(), dark)
        self.settings.setValue("dark", dark)
        self.pipeline.apply_theme()
        self.pipeline.on_state()
        self.perf.request()
        self.ofdm.request()
        self.update()

    def _tab_changed(self, idx: int) -> None:
        self.pipeline.set_active(self.tabs.widget(idx) is self.pipeline)

    # ---------------------------------------------------------------- ações
    def open_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Abrir configuração", "", "Configuração (*.json)")
        if not path:
            return
        try:
            cfg, ch, cn = config_io.load(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Erro", f"Arquivo inválido:\n{exc}")
            return
        self.panel.load_state(cfg, ch, cn)

    def save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Salvar configuração", "config_isdbt.json",
                                              "Configuração (*.json)")
        if path:
            config_io.save(path, self.state.cfg, self.state.channel_name, self.state.cn_db)

    def export_report(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Exportar relatório", "relatorio_isdbt.pdf", "PDF (*.pdf)")
        if not path:
            return
        from ..core.report import export_pdf
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            export_pdf(path, self.state.cfg, self.state.table_channel() or "AWGN", self.state.cn_db, __author__)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Erro", str(exc))
        finally:
            QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"Relatório salvo em {path}", 6000)

    def about(self) -> None:
        QMessageBox.about(
            self, f"Sobre o {__app_name__}",
            f"<h3>{__app_name__} {__version__}</h3>"
            f"<p>Previsão de desempenho e simulação didática do ISDB-T / ISDB-Tb.</p>"
            f"<p>Desenvolvido por <b>{__author__}</b> · ObelLab</p>"
            "<p>Python · PySide6 (Qt) · NumPy · SciPy · Matplotlib · pyqtgraph · Pillow · OpenCV · FFmpeg</p>"
            f"<p>Licença GPL-3.0 · <a href='{__url__}'>{__url__}</a></p>")

    def closeEvent(self, ev) -> None:  # noqa: N802
        self.settings.setValue("geometry", self.saveGeometry())
        self.pipeline.shutdown()
        self.video.cleanup()
        super().closeEvent(ev)


def run() -> int:
    multiprocessing.freeze_support()
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("ISDB-T Analyzer")
    app.setOrganizationName("ObelLab")
    app.setDesktopFileName("isdbt-analyzer")
    app.setWindowIcon(_icon())
    dark = QSettings("ObelLab", "ISDBT-Analyzer").value("dark", True, type=bool)
    theme.apply(app, dark)
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=False)  # desenho mais leve; as constelações usam mapa de densidade
    win = MainWindow(dark)
    win.show()
    return app.exec()
