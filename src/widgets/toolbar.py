from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
)


from .utils import get_asset_path


class ToolBar(QWidget):
    """Floating toolbar on the right with tools only."""

    hand = Signal()
    pen = Signal()
    eraser = Signal()
    join = Signal()
    save = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.selected_tool = None

        toolsLayout = QVBoxLayout()
        toolsLayout.setContentsMargins(3, 3, 3, 3)
        toolsLayout.setSpacing(2)

        self.handButton = QPushButton()
        self.handButton.setIcon(QIcon(str(get_asset_path("icons/move.png"))))
        self.handButton.setIconSize(QSize(40, 40))

        self.penButton = QPushButton()
        self.penButton.setIcon(QIcon(str(get_asset_path("icons/marker.png"))))
        self.penButton.setIconSize(QSize(40, 40))

        self.eraseButton = QPushButton()
        self.eraseButton.setIcon(QIcon(str(get_asset_path("icons/clean.png"))))
        self.eraseButton.setIconSize(QSize(40, 40))

        self.joinButton = QPushButton()
        self.joinButton.setIcon(QIcon(str(get_asset_path("icons/join.png"))))
        self.joinButton.setIconSize(QSize(40, 40))

        self.saveButton = QPushButton()
        self.saveButton.setIcon(QIcon(str(get_asset_path("icons/save.png"))))
        self.saveButton.setIconSize(QSize(40, 40))

        font = QFont()
        font.setPointSize(20)
        for btn in [
            self.handButton,
            self.penButton,
            self.eraseButton,
            self.joinButton,
            self.saveButton,
        ]:
            btn.setMinimumWidth(70)
            btn.setMinimumHeight(70)
            btn.setMaximumWidth(70)
            btn.setMaximumHeight(70)
            btn.setFont(font)

        self.handButton.clicked.connect(self._on_hand_clicked)
        self.penButton.clicked.connect(self._on_pen_clicked)
        self.eraseButton.clicked.connect(self._on_eraser_clicked)
        self.joinButton.clicked.connect(self._on_join_clicked)
        self.saveButton.clicked.connect(self._on_save_clicked)

        toolsLayout.addWidget(self.handButton)
        toolsLayout.addWidget(self.penButton)
        toolsLayout.addWidget(self.eraseButton)
        toolsLayout.addWidget(self.joinButton)
        toolsLayout.addWidget(self.saveButton)

        self._highlight_style = "background-color: #4CAF50; font-weight: bold;"
        self._normal_style = ""

        self.setLayout(toolsLayout)
        self._set_button_highlight(self.handButton)

        self.setMinimumWidth(90)
        self.setMaximumWidth(90)
        self.adjustSize()

    def _set_button_highlight(self, button):
        """Highlight the selected tool button"""
        if self.selected_tool:
            self.selected_tool.setStyleSheet(self._normal_style)

        button.setStyleSheet(self._highlight_style)
        self.selected_tool = button

    def _on_hand_clicked(self):
        self._set_button_highlight(self.handButton)
        self.hand.emit()

    def _on_pen_clicked(self):
        self._set_button_highlight(self.penButton)
        self.pen.emit()

    def _on_eraser_clicked(self):
        self._set_button_highlight(self.eraseButton)
        self.eraser.emit()

    def _on_join_clicked(self):
        self._set_button_highlight(self.joinButton)
        self.join.emit()

    def _on_save_clicked(self):
        self.save.emit()
