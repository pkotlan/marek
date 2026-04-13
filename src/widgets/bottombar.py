from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from .utils import get_asset_path


class BottomBar(QWidget):
    """Floating bottom bar with image navigation and counter."""

    open_image_clicked = Signal()
    nextImage = Signal()
    prevImage = Signal()
    validate_clicked = Signal()
    export_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(5)

        prevButton = QPushButton()
        prevButton.setIcon(QIcon(str(get_asset_path("icons/arrow-left.png"))))
        prevButton.setIconSize(QSize(40, 40))
        prevButton.setMinimumSize(60, 45)
        prevButton.setMaximumSize(60, 45)
        prevButton.clicked.connect(self.prevImage.emit)
        layout.addWidget(prevButton)

        layout.addSpacing(10)

        self.counterLabel = QLabel("No images loaded")
        self.counterLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.counterLabel.setMinimumWidth(140)
        self.counterLabel.setAutoFillBackground(True)
        layout.addWidget(self.counterLabel)

        layout.addSpacing(10)

        openButton = QPushButton()
        openButton.setIcon(QIcon(str(get_asset_path("icons/add-image.png"))))
        openButton.setIconSize(QSize(40, 40))
        openButton.setMinimumSize(60, 45)
        openButton.setMaximumSize(60, 45)
        openButton.clicked.connect(self.open_image_clicked.emit)
        layout.addWidget(openButton)

        layout.addSpacing(10)

        self.validateButton = QPushButton()
        self.validateButton.setIcon(QIcon(str(get_asset_path("icons/check.png"))))
        self.validateButton.setIconSize(QSize(40, 40))
        self.validateButton.setMinimumSize(60, 45)
        self.validateButton.setMaximumSize(60, 45)
        self.validateButton.clicked.connect(self.validate_clicked.emit)
        layout.addWidget(self.validateButton)

        layout.addSpacing(10)

        self.exportButton = QPushButton()
        self.exportButton.setIcon(QIcon(str(get_asset_path("icons/export.png"))))
        self.exportButton.setIconSize(QSize(40, 40))
        self.exportButton.setMinimumSize(60, 45)
        self.exportButton.setMaximumSize(60, 45)
        self.exportButton.clicked.connect(self.export_clicked.emit)
        layout.addWidget(self.exportButton)

        layout.addSpacing(10)

        nextButton = QPushButton()
        nextButton.setIcon(QIcon(str(get_asset_path("icons/arrow-right.png"))))
        nextButton.setIconSize(QSize(40, 40))
        nextButton.setMinimumSize(60, 45)
        nextButton.setMaximumSize(60, 45)
        nextButton.clicked.connect(self.nextImage.emit)
        layout.addWidget(nextButton)

        self.setLayout(layout)
        self.setMinimumHeight(50)
        self.setMaximumHeight(50)
        self.adjustSize()

    def update_counter(self, current: int, total: int):
        """Update the image counter display."""
        if total == 0:
            self.counterLabel.setText("No images loaded")
        else:
            self.counterLabel.setText(f"Image {current + 1} of {total}")
