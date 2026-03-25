import math
from enum import StrEnum
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import QWidget
from skimage.draw import polygon as draw_polygon
from skimage.measure import find_contours

CLOSE_THRESHOLD = 30
MIN_POINT_DISTANCE = 1
COLORS = [
    QColor(255, 0, 0),
    QColor(0, 255, 0),
    QColor(0, 0, 255),
    QColor(255, 255, 0),
    QColor(255, 0, 255),
]


class Tool(StrEnum):
    HAND = "hand"
    PEN = "pen"
    ERASER = "eraser"


class Canvas(QWidget):
    objects_updated = Signal()

    def __init__(self):
        super().__init__()
        self.image = None
        self.image_path = None
        self.zoom = 1.0
        self.offset = QPointF(0, 0)
        self.drawing = False

        self.current_points: list[QPointF] = []
        self.objects: list[list[QPointF]] = []

        self.pan_start = QPoint(0, 0)
        self.tool: Tool = Tool.HAND
        self.scaled_image_cache = None
        self.cached_zoom = 1.0

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    def load_image(self, file_path, objects: list[list[QPointF]] | None = None):
        self.image_path = file_path
        self.image = QImage(file_path)
        self.zoom = 1.0
        self.offset = QPointF(0, 0)
        self.scaled_image_cache = self.image.copy()
        self.cached_zoom = self.zoom

        if objects is None:
            objects = self._load_objects_from_npy(file_path)

        self.objects = objects if objects else []
        self.current_points = []
        self.fit_to_window()
        self.update()

    def _load_objects_from_npy(self, file_path: str) -> list[list[QPointF]] | None:
        from pathlib import Path

        import numpy as np
        from scipy.interpolate import splev, splprep

        npy_path = Path(file_path).with_suffix(".npy")
        if not npy_path.exists():
            return None

        try:
            labels = np.load(npy_path, allow_pickle=True)
            if isinstance(labels, np.ndarray) and labels.dtype == object:
                labels = labels.item()
            if isinstance(labels, dict):
                labels = labels.get("labels", labels)

            objects = []

            for label_num in np.unique(labels):
                if label_num == 0:
                    continue

                mask = (labels == label_num).astype(float)
                contours = find_contours(mask, 0.5)

                if contours:
                    contour = max(contours, key=len)

                    y, x = contour[:, 0], contour[:, 1]  # ty:ignore[not-subscriptable]

                    if len(x) > 4:
                        if not np.allclose([x[0], y[0]], [x[-1], y[-1]]):
                            x = np.append(x, x[0])
                            y = np.append(y, y[0])

                        tck, u = splprep([x, y], s=3.0, per=True)

                        u_new = np.linspace(u.min(), u.max(), len(x))
                        x_new, y_new = splev(u_new, tck)

                        qpoints = [QPointF(nx, ny) for nx, ny in zip(x_new, y_new)]
                    else:
                        qpoints = [QPointF(nx, ny) for nx, ny in zip(x, y)]

                    if len(qpoints) >= 3:
                        objects.append(qpoints)

            return objects if objects else None
        except Exception as e:
            print(f"Error loading objects from {npy_path}: {e}")
            return None

    def fit_to_window(self):
        if not self.image:
            return

        window_width = self.width()
        window_height = self.height()
        img_width = self.image.width()
        img_height = self.image.height()

        if window_width > 0 and window_height > 0:
            zoom_x = window_width / img_width
            zoom_y = window_height / img_height
            self.zoom = min(zoom_x, zoom_y) * 0.9
            self.center_image()

    def center_image(self):
        if not self.image:
            return

        img_width = self.image.width() * self.zoom
        img_height = self.image.height() * self.zoom

        x = (self.width() - img_width) / 2.0
        y = (self.height() - img_height) / 2.0

        self.offset = QPointF(x, y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), self.palette().color(QPalette.ColorRole.Window))

        if not self.image:
            return

        if self.cached_zoom != self.zoom:
            mode = (
                Qt.TransformationMode.FastTransformation
                if self.drawing
                else Qt.TransformationMode.SmoothTransformation
            )
            self.scaled_image_cache = self.image.scaledToWidth(
                int(self.image.width() * self.zoom),
                mode,
            )
            self.cached_zoom = self.zoom

        if not self.scaled_image_cache:
            return

        painter.drawImage(
            int(self.offset.x()), int(self.offset.y()), self.scaled_image_cache
        )

        for i, obj in enumerate(self.objects):
            base_color = COLORS[i % len(COLORS)]
            painter.setPen(QPen(base_color, 2, Qt.PenStyle.SolidLine))
            fill_color = QColor(base_color)
            fill_color.setAlpha(100)
            painter.setBrush(QBrush(fill_color))
            self._draw_polygon(painter, obj, closed=True)

        if self.current_points:
            color = COLORS[len(self.objects) % len(COLORS)]
            painter.setPen(QPen(color, 2, Qt.PenStyle.SolidLine))
            self._draw_polygon(painter, self.current_points, closed=False)

    def _draw_polygon(self, painter, points, closed=True):
        if len(points) < 2:
            return

        screen_points = [self.screen_coords(p) for p in points]
        path = self._create_path(screen_points, closed=closed)

        if not closed:
            painter.setBrush(Qt.BrushStyle.NoBrush)

        painter.drawPath(path)

    def _create_path(self, points, closed=False):
        path = QPainterPath()
        if len(points) < 2:
            return path

        path.moveTo(points[0])
        for p in points[1:]:
            path.lineTo(p)

        if closed:
            path.closeSubpath()
        return path

    def _polygon_contains(self, polygon, point):
        path = QPainterPath()
        path.addPolygon([QPointF(p.x(), p.y()) for p in polygon])
        return path.contains(QPointF(point.x(), point.y()))

    def screen_coords(self, image_point):
        x = image_point.x() * self.zoom + self.offset.x()
        y = image_point.y() * self.zoom + self.offset.y()
        return QPointF(x, y)

    def image_coords(self, screen_point):
        if not self.image:
            return QPointF(0, 0)

        x = (screen_point.x() - self.offset.x()) / self.zoom
        y = (screen_point.y() - self.offset.y()) / self.zoom

        return QPointF(
            max(0.0, min(x, float(self.image.width() - 1))),
            max(0.0, min(y, float(self.image.height() - 1))),
        )

    def wheelEvent(self, event):
        if not self.image:
            return

        global_pos = event.globalPosition()
        mouse_screen = self.mapFromGlobal(
            QPoint(int(global_pos.x()), int(global_pos.y()))
        )
        mouse_image = self.image_coords(mouse_screen)

        zoom_factor = 1.1
        if event.angleDelta().y() > 0:
            self.zoom *= zoom_factor
        else:
            self.zoom /= zoom_factor

        self.zoom = max(0.1, min(self.zoom, 10.0))

        self.offset.setX(mouse_screen.x() - (mouse_image.x() * self.zoom))
        self.offset.setY(mouse_screen.y() - (mouse_image.y() * self.zoom))

        self.update()

    def mousePressEvent(self, event):
        if not self.image:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            match self.tool:
                case Tool.HAND:
                    self.pan_start = event.pos()
                case Tool.PEN:
                    click_pos = self.image_coords(event.pos())
                    self.drawing = True
                    self.current_points.append(click_pos)
                    self.update()
                case Tool.ERASER:
                    click_pos = self.image_coords(event.pos())
                    self.objects = [
                        obj
                        for obj in self.objects
                        if not self._polygon_contains(obj, click_pos)
                    ]
                    self.objects_updated.emit()
                    self.update()

    def mouseMoveEvent(self, event):
        if not self.image:
            return

        if event.buttons() & Qt.MouseButton.LeftButton:
            match self.tool:
                case Tool.HAND:
                    delta = QPointF(event.pos() - self.pan_start)
                    self.offset += delta
                    self.pan_start = event.pos()
                    self.update()
                case Tool.PEN:
                    if self.drawing:
                        new_point = self.image_coords(event.pos())

                        if self.current_points:
                            last = self.current_points[-1]
                            dist = math.hypot(
                                new_point.x() - last.x(), new_point.y() - last.y()
                            )
                            if dist >= MIN_POINT_DISTANCE:
                                self.current_points.append(new_point)
                                self.update()
                        else:
                            self.current_points.append(new_point)
                            self.update()

    def mouseReleaseEvent(self, event):
        if not self.image:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

            if len(self.current_points) >= 3:
                start = self.screen_coords(self.current_points[0])
                end = event.pos()
                distance = math.hypot(end.x() - start.x(), end.y() - start.y())

                if distance <= CLOSE_THRESHOLD:
                    self.objects.append(self.current_points)
                    self.objects_updated.emit()
                    self.current_points = []

            self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image:
            self.fit_to_window()

    def set_tool_hand(self):
        self.tool = Tool.HAND

    def set_tool_pen(self):
        self.tool = Tool.PEN

    def set_tool_eraser(self):
        self.tool = Tool.ERASER

    def save(self):
        if not (self.image_path and self.image) or not self.objects:
            return

        labels = np.zeros((self.image.height(), self.image.width()), dtype=np.uint16)

        for label_num, obj in enumerate(self.objects, start=1):
            if len(obj) >= 3:
                rows = [p.y() for p in obj]
                cols = [p.x() for p in obj]
                rr, cc = draw_polygon(rows, cols, labels.shape)
                labels[rr, cc] = label_num

        save_path = Path(self.image_path).with_suffix(".npy")

        np.save(save_path, labels)
        print(f"Mask saved pure to {save_path}")
