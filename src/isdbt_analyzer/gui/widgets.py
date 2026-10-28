"""Widgets utilitários."""

from __future__ import annotations

import numpy as np
from PIL import Image
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from . import theme


def pil_to_qimage(img: Image.Image) -> QImage:
    img = img.convert("RGB")
    arr = np.ascontiguousarray(np.asarray(img))
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class ImageView(QWidget):
    """Mostra uma imagem ajustada ao espaço, sem distorcer, com legenda."""

    def __init__(self, caption: str = "") -> None:
        super().__init__()
        self.caption = caption
        self.pix: QPixmap | None = None
        self.setMinimumSize(240, 160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, img: QImage | None) -> None:
        self.pix = QPixmap.fromImage(img) if img is not None else None
        self.update()

    def set_caption(self, text: str) -> None:
        self.caption = text
        self.update()

    def paintEvent(self, _ev) -> None:  # noqa: N802
        c = theme.colors()
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(c["plot_bg"]))
        top = 22
        area = QRectF(0, top, self.width(), self.height() - top)
        if self.pix and not self.pix.isNull():
            s = min(area.width() / self.pix.width(), area.height() / self.pix.height())
            w, h = self.pix.width() * s, self.pix.height() * s
            target = QRectF(area.x() + (area.width() - w) / 2, area.y() + (area.height() - h) / 2, w, h)
            p.setRenderHint(QPainter.SmoothPixmapTransform, s < 1)
            p.drawPixmap(target, self.pix, QRectF(self.pix.rect()))
        else:
            p.setPen(QColor(c["muted"]))
            p.drawText(area, Qt.AlignCenter, "sem imagem")
        p.setPen(QColor(c["text"]))
        p.drawText(QRectF(6, 2, self.width() - 12, top - 4), Qt.AlignLeft | Qt.AlignVCenter, self.caption)
        p.end()
