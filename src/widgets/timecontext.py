"""Read-only temporal image preview, independent of the annotation canvas."""

import zipfile

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class ImageView(QGraphicsView):
    """
    display, zoom, pan, resize-to-fit
    """

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        # Avoid fitInView triggering another resize as scrollbars appear/disappear.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.pixmap = self.scene().addPixmap(QPixmap())
        self.image_size = None

    def show_image(self, image):
        size = image.size()
        refit = size != self.image_size
        self.pixmap.setPixmap(QPixmap.fromImage(image))
        self.setSceneRect(0, 0, image.width(), image.height())
        self.image_size = size
        if refit:
            self.fit()

    def fit(self):
        if self.image_size is not None:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if self.image_size is not None:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            if 0.02 <= self.transform().m11() * factor <= 30:
                self.scale(factor, factor)
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()


class Timeline(QSlider):
    def __init__(self):
        super().__init__(Qt.Orientation.Horizontal)
        self.offsets = [0]
        self.setMinimumHeight(52)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: palette(mid);
            }
            QSlider::handle:horizontal {
                width: 26px;
                margin: -9px 0;
                background: palette(highlight);
            }
        """)

    def paintEvent(self, event):
        super().paintEvent(event)
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        selected = min(self.offsets, key=lambda n: (abs(n - self.value()), abs(n)))
        for offset in self.offsets:
            option.sliderPosition = offset
            option.sliderValue = offset
            handle = self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider,
                option,
                QStyle.SubControl.SC_SliderHandle,
                self,
            )
            color = (
                self.palette().highlight().color()
                if offset == selected and self.isEnabled()
                else self.palette().text().color()
            )
            painter.setPen(
                QPen(color, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            )
            x = handle.center().x()
            bottom = min(handle.bottom() + 8, self.height() - 3)
            painter.drawLine(x, bottom - (6 if offset == 0 else 3), x, bottom)


class TimeContextPreview(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setFont(QFont(self.font().family(), 13))
        self.resize(960, 640)
        self.frames = {}
        self.cache = {}
        self.zip_path = ""
        self.selected = 0
        self.view = ImageView()
        self.message = QLabel(self.view.viewport())
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setMaximumWidth(600)
        self.message.setMinimumWidth(0)
        self.message.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.message.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.message.setFont(QFont(self.font().family(), 16, QFont.Weight.Bold))
        self.message.setStyleSheet(
            "color: white; background: rgba(20, 24, 32, 225); padding: 16px;"
        )
        overlay = QVBoxLayout(self.view.viewport())
        overlay.setContentsMargins(24, 24, 24, 24)
        overlay.addStretch()
        message_row = QHBoxLayout()
        message_row.addStretch(1)
        message_row.addWidget(self.message, 4)
        message_row.addStretch(1)
        overlay.addLayout(message_row)
        overlay.addStretch()
        self.message.hide()
        self.time_label = QLabel()
        self.time_label.setFont(QFont(self.font().family(), 17, QFont.Weight.Bold))
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider = Timeline()
        self.slider.setAccessibleName("Time offset")
        self.slider.valueChanged.connect(self._select)
        self.slider.sliderReleased.connect(self._snap)
        self.earlier = QToolButton()
        self.later = QToolButton()
        for button, arrow, text in (
            (self.earlier, Qt.ArrowType.LeftArrow, "Earlier frame"),
            (self.later, Qt.ArrowType.RightArrow, "Later frame"),
        ):
            button.setArrowType(arrow)
            button.setFixedSize(40, 40)
            button.setToolTip(text)
            button.setAccessibleName(text)
            button.setEnabled(False)
        self.earlier.clicked.connect(lambda: self._step(-1))
        self.later.clicked.connect(lambda: self._step(1))
        slider_row = QHBoxLayout()
        slider_row.addWidget(self.earlier)
        slider_row.addWidget(self.slider, 1)
        slider_row.addWidget(self.later)
        fit = QPushButton("Fit")
        fit.clicked.connect(self.view.fit)
        timeline = QWidget()
        timeline.setMaximumWidth(520)
        timeline_layout = QVBoxLayout(timeline)
        timeline_layout.setContentsMargins(12, 0, 12, 0)
        timeline_layout.setSpacing(2)
        timeline_layout.addWidget(self.time_label)
        timeline_layout.addLayout(slider_row)
        timeline_row = QHBoxLayout()
        timeline_row.addStretch()
        timeline_row.addWidget(timeline, 1)
        timeline_row.addStretch()
        layout = QVBoxLayout(self)
        layout.addWidget(self.view, 1)
        layout.addLayout(timeline_row)
        layout.addWidget(fit, alignment=Qt.AlignmentFlag.AlignRight)

    def set_item(self, zip_path, center, frames):
        self.zip_path = zip_path
        self.frames = frames
        self.cache.clear()
        self.selected = 0
        self.view.image_size = None
        self.setWindowTitle(f"{center} — Time context · Read-only")
        limit = max(abs(n) for n in frames)
        self.slider.blockSignals(True)
        self.slider.offsets = sorted(frames)
        self.slider.setRange(-limit, limit)
        self.slider.setValue(0)
        self.slider.setEnabled(len(frames) > 1)
        self.slider.blockSignals(False)
        self._select(0)

    def _snap(self):
        self.slider.setValue(self.selected)

    def _step(self, direction):
        index = self.slider.offsets.index(self.selected)
        self.slider.setValue(self.slider.offsets[index + direction])

    def _set_message(self, text):
        self.message.setText(text)
        self.message.setVisible(bool(text))
        self.message.raise_()

    def _select(self, value):
        offset = min(self.frames, key=lambda n: (abs(n - value), abs(n)))
        if offset == self.selected and offset in self.cache:
            return
        self.selected = offset
        self.earlier.setEnabled(offset > self.slider.offsets[0])
        self.later.setEnabled(offset < self.slider.offsets[-1])
        self.time_label.setText(
            "0 s · annotation frame" if offset == 0 else f"{offset / 1000:+g} s"
        )
        self._set_message(
            "No context frames available" if len(self.frames) == 1 else ""
        )
        if offset not in self.cache:
            with zipfile.ZipFile(self.zip_path) as archive:
                self.cache[offset] = QImage.fromData(archive.read(self.frames[offset]))
        self.view.show_image(self.cache[offset])
