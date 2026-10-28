"""Tema visual (escuro por padrão, estilo instrumento de medição) e cores compartilhadas."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

ACCENT = "#4fb3bf"
GOOD = "#3fb950"
WARN = "#d29922"
BAD = "#e5484d"

DARK = {
    "bg": "#16181d", "panel": "#1e2128", "panel2": "#262a33", "border": "#343944",
    "text": "#e6e6e6", "muted": "#9aa3ae", "plot_bg": "#111318", "grid": "#2c313b",
}
LIGHT = {
    "bg": "#f4f5f7", "panel": "#ffffff", "panel2": "#eef0f3", "border": "#d0d5dd",
    "text": "#1f2328", "muted": "#59636e", "plot_bg": "#ffffff", "grid": "#e1e4e8",
}

_current = DARK


def colors() -> dict:
    return _current


def is_dark() -> bool:
    return _current is DARK


def health_color(margin_db: float | None) -> str:
    if margin_db is None:
        return colors()["muted"]
    if margin_db >= 3:
        return GOOD
    if margin_db >= 0:
        return WARN
    return BAD


def apply(app: QApplication, dark: bool = True) -> None:
    global _current
    _current = DARK if dark else LIGHT
    c = _current
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(c["bg"]))
    pal.setColor(QPalette.WindowText, QColor(c["text"]))
    pal.setColor(QPalette.Base, QColor(c["panel"]))
    pal.setColor(QPalette.AlternateBase, QColor(c["panel2"]))
    pal.setColor(QPalette.Text, QColor(c["text"]))
    pal.setColor(QPalette.Button, QColor(c["panel2"]))
    pal.setColor(QPalette.ButtonText, QColor(c["text"]))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("#0b0d10"))
    pal.setColor(QPalette.ToolTipBase, QColor(c["panel2"]))
    pal.setColor(QPalette.ToolTipText, QColor(c["text"]))
    pal.setColor(QPalette.PlaceholderText, QColor(c["muted"]))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(c["muted"]))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(c["muted"]))
    app.setPalette(pal)
    app.setStyleSheet(f"""
        QMainWindow, QWidget#root {{ background: {c['bg']}; }}
        QGroupBox {{
            border: 1px solid {c['border']}; border-radius: 8px; margin-top: 14px;
            padding: 10px 8px 8px 8px; background: {c['panel']}; font-weight: 600;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {c['muted']}; }}
        QTabWidget::pane {{ border: 1px solid {c['border']}; border-radius: 8px; background: {c['panel']}; top: -1px; }}
        QTabBar::tab {{
            background: transparent; color: {c['muted']}; padding: 8px 14px; border: none;
            border-bottom: 2px solid transparent; font-weight: 600;
        }}
        QTabBar::tab:selected {{ color: {c['text']}; border-bottom: 2px solid {ACCENT}; }}
        QTabBar::tab:hover {{ color: {c['text']}; }}
        QPushButton {{
            background: {c['panel2']}; border: 1px solid {c['border']}; border-radius: 6px;
            padding: 6px 12px;
        }}
        QPushButton:hover {{ border-color: {ACCENT}; }}
        QPushButton#primary {{ background: {ACCENT}; color: #0b0d10; border: none; font-weight: 700; }}
        QPushButton:disabled {{ color: {c['muted']}; }}
        QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
            background: {c['panel2']}; border: 1px solid {c['border']}; border-radius: 5px; padding: 3px 6px;
        }}
        QSlider::groove:horizontal {{ height: 6px; background: {c['border']}; border-radius: 3px; }}
        QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
        QSlider::handle:horizontal {{
            background: {c['text']}; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px;
        }}
        QLabel#muted {{ color: {c['muted']}; }}
        QLabel#kpiValue {{ font-size: 20px; font-weight: 700; }}
        QLabel#kpiTitle {{ color: {c['muted']}; font-size: 11px; }}
        QFrame#kpi {{ background: {c['panel2']}; border: 1px solid {c['border']}; border-radius: 8px; }}
        QTableWidget {{ gridline-color: {c['border']}; border: none; }}
        QHeaderView::section {{
            background: {c['panel2']}; color: {c['muted']}; border: none; padding: 4px; font-weight: 600;
        }}
        QStatusBar {{ color: {c['muted']}; }}
        QProgressBar {{ border: 1px solid {c['border']}; border-radius: 5px; text-align: center; }}
        QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
        QScrollArea {{ border: none; }}
    """)
