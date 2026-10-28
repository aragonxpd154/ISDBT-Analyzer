"""Teste de fumaça da interface (roda sem tela, com o plugin offscreen do Qt)."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    from isdbt_analyzer.gui import theme
    qa = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(qa, True)
    return qa


def test_main_window_runs(app):
    from isdbt_analyzer.gui.app import MainWindow
    w = MainWindow(True)
    w.show()
    try:
        w.panel.cn_spin.setValue(12.0)
        w.panel.channel.setCurrentText("Brasil A")
        deadline = time.time() + 20
        while time.time() < deadline and w.pipeline.acc["packets"] == 0:
            app.processEvents()
            w.pipeline.tick()
            time.sleep(0.05)
        assert w.pipeline.acc["packets"] > 0          # o processo de FEC respondeu
        assert w.pipeline.buf_eq.size > 0             # constelações recebendo símbolos
        for i in range(w.tabs.count()):
            w.tabs.setCurrentIndex(i)
            app.processEvents()
    finally:
        w.close()
