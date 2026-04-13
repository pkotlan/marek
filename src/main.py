import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
)

from widgets.bottombar import BottomBar
from widgets.canvas import Canvas
from widgets.toolbar import ToolBar


def exception_hook(exc_type, exc_value, exc_traceback):
    logging.error("Unhandled exception:", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = exception_hook


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAReK")
        self.setGeometry(100, 100, 1200, 800)

        self.zip_path: str = ""
        self.npz_path: str = ""
        self.image_names: list[str] = []
        self.npz_data: dict = {}
        self.currImgIdx: int = 0
        self.objects_map: dict[str, list] = {}

        self.canvas = Canvas()
        self.setCentralWidget(self.canvas)

        self.toolbar = ToolBar(self.canvas)
        self.bottom_bar = BottomBar(self.canvas)

        self.bottom_bar.open_image_clicked.connect(self.open_dataset)
        self.toolbar.hand.connect(self.canvas.set_tool_hand)
        self.toolbar.pen.connect(self.canvas.set_tool_pen)
        self.toolbar.eraser.connect(self.canvas.set_tool_eraser)
        self.toolbar.save.connect(self.save_to_npz)
        self.toolbar.join.connect(self.canvas.set_tool_join)

        self.bottom_bar.nextImage.connect(self.next_image)
        self.bottom_bar.prevImage.connect(self.prev_image)
        self.canvas.objects_updated.connect(self.update_objects_map)

        self.toolbar.show()
        self.bottom_bar.show()

        self._position_floating_panels()

    def open_dataset(self):
        zip_file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image Dataset",
            "",
            "Zip Archives (*.zip)",
        )

        if not zip_file_path:
            return

        zip_path_obj = Path(zip_file_path)
        npz_file_path = zip_path_obj.with_suffix(".npz")

        if not npz_file_path.exists():
            QMessageBox.critical(
                self,
                "Missing Annotation File",
                f"Could not find matching label file:\n{npz_file_path.name}\n\nPlease ensure both the .zip and .npz share the same name and are in the same directory.",
            )
            return

        try:
            self.npz_data = dict(np.load(npz_file_path, allow_pickle=True))
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load .npz file:\n{e}")
            return

        try:
            with zipfile.ZipFile(zip_file_path, "r") as zf:
                self.image_names = [
                    name
                    for name in zf.namelist()
                    if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
                ]
                self.image_names.sort()
        except Exception as e:
            QMessageBox.critical(
                self, "Zip Error", f"Failed to read the .zip file:\n{e}"
            )
            return

        if not self.image_names:
            QMessageBox.warning(
                self,
                "Empty Dataset",
                "No valid images found in the selected .zip file.",
            )
            return

        self.zip_path = zip_file_path
        self.npz_path = str(npz_file_path)
        self.currImgIdx = 0
        self.objects_map.clear()

        self._load_current_image()

    def next_image(self):
        if not self.image_names:
            return
        self.currImgIdx = (self.currImgIdx + 1) % len(self.image_names)
        self._load_current_image()

    def prev_image(self):
        if not self.image_names:
            return
        self.currImgIdx = (self.currImgIdx - 1) % len(self.image_names)
        self._load_current_image()

    def _load_current_image(self):
        """Extracts current image from zip, finds its labels, and loads them into canvas."""
        if not self.image_names:
            return

        img_name = self.image_names[self.currImgIdx]

        with zipfile.ZipFile(self.zip_path, "r") as zf:
            img_data = zf.read(img_name)

        qimage = QImage()
        qimage.loadFromData(img_data)

        base_name = Path(img_name).stem

        labels_array = None
        if img_name in self.npz_data:
            labels_array = self.npz_data[img_name]
        elif base_name in self.npz_data:
            labels_array = self.npz_data[base_name]
        elif f"{base_name}.npy" in self.npz_data:
            labels_array = self.npz_data[f"{base_name}.npy"]

        objects = self.objects_map.get(img_name)

        self.canvas.load_dataset_item(img_name, qimage, labels_array, objects)
        self.bottom_bar.update_counter(self.currImgIdx, len(self.image_names))

    def update_objects_map(self):
        if not self.image_names:
            return
        current_name = self.image_names[self.currImgIdx]
        if self.canvas.objects:
            self.objects_map[current_name] = self.canvas.objects

    def _position_floating_panels(self):
        canvas_width = self.canvas.width()
        canvas_height = self.canvas.height()

        toolbar_width = self.toolbar.width()
        toolbar_height = self.toolbar.height()
        toolbar_x = canvas_width - toolbar_width - 20
        toolbar_y = (canvas_height - toolbar_height) // 2
        self.toolbar.move(toolbar_x, toolbar_y)
        self.toolbar.raise_()

        bottom_width = self.bottom_bar.width()
        bottom_height = self.bottom_bar.height()
        bottom_x = (canvas_width - bottom_width) // 2
        bottom_y = canvas_height - bottom_height - 20
        self.bottom_bar.move(bottom_x, bottom_y)
        self.bottom_bar.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_floating_panels()

    def moveEvent(self, event):
        super().moveEvent(event)
        self._position_floating_panels()

    def save_to_npz(self):
        if not self.image_names or not self.npz_path:
            return

        self.update_objects_map()

        updated_data = dict(self.npz_data)

        temp_canvas = Canvas()
        if self.canvas.image:
            temp_canvas.image = self.canvas.image

        for img_path, objects_list in self.objects_map.items():
            temp_canvas.objects = objects_list
            mask = temp_canvas.get_current_mask()

            if mask is not None:
                updated_data[img_path] = mask

        try:
            np.savez_compressed(self.npz_path, **updated_data)
            self.npz_data = updated_data
            QMessageBox.information(
                self,
                "Success",
                f"Successfully updated labels in\n{Path(self.npz_path).name}",
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Save Error", f"Failed to overwrite .npz file:\n{e}"
            )


app = QApplication()

# stylesheet_path = Path(__file__).parent.parent / "assets" / "styles.qss"
# with open(stylesheet_path, "r") as f:
#     app.setStyleSheet(f.read())

window = MainWindow()
window.show()


if __name__ == "__main__":
    _ = app.exec()
