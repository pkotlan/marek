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
    JOIN = "join"


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

        self.join_base_idx = -1

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    def load_dataset_item(
        self,
        image_internal_path: str,
        qimage: QImage,
        labels_array: np.ndarray | None,
        objects: list[list[QPointF]] | None = None,
    ):
        self.image_path = image_internal_path
        self.image = qimage
        self.zoom = 1.0
        self.offset = QPointF(0, 0)
        self.scaled_image_cache = self.image.copy()
        self.cached_zoom = self.zoom

        if objects is not None:
            self.objects = objects
        elif labels_array is not None:
            self.objects = self._extract_objects_from_labels(labels_array)
        else:
            self.objects = []

        self._reset_interaction_state()

        self.fit_to_window()
        self.update()

    def _extract_objects_from_labels(self, labels) -> list[list[QPointF]]:
        from scipy.interpolate import splev, splprep

        if isinstance(labels, np.ndarray) and labels.dtype == object:
            labels = labels.item()
        if isinstance(labels, dict):
            labels = labels.get("labels", labels)

        objects = []
        try:
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
        except Exception as e:
            print(f"Error processing mask data: {e}")

        return objects

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

    def paintEvent(self, _):
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
            if self.tool == Tool.JOIN and i == self.join_base_idx:
                base_color = QColor(255, 255, 255)
            else:
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
                case Tool.JOIN:
                    click_pos = self.image_coords(event.pos())
                    clicked_idx = -1

                    for i, obj in enumerate(self.objects):
                        if self._polygon_contains(obj, click_pos):
                            clicked_idx = i
                            break

                    if clicked_idx == -1:
                        self.join_base_idx = -1
                    else:
                        if self.join_base_idx == -1:
                            self.join_base_idx = clicked_idx
                        elif self.join_base_idx != clicked_idx:
                            base_obj = self.objects[self.join_base_idx]
                            clicked_obj = self.objects[clicked_idx]

                            mask = np.zeros(
                                (self.image.height(), self.image.width()),
                                dtype=np.uint8,
                            )

                            for obj in [base_obj, clicked_obj]:
                                rows = [
                                    max(0, min(self.image.height() - 1, p.y()))
                                    for p in obj
                                ]
                                cols = [
                                    max(0, min(self.image.width() - 1, p.x()))
                                    for p in obj
                                ]
                                rr, cc = draw_polygon(rows, cols, mask.shape)
                                mask[rr, cc] = 1

                            min_dist = float("inf")
                            best_p1, best_p2 = None, None
                            for p1 in base_obj:
                                for p2 in clicked_obj:
                                    dist = math.hypot(p1.x() - p2.x(), p1.y() - p2.y())
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_p1 = p1
                                        best_p2 = p2

                            if best_p1 and best_p2:
                                dx = best_p2.x() - best_p1.x()
                                dy = best_p2.y() - best_p1.y()
                                length = math.hypot(dx, dy)
                                if length > 0:
                                    nx = -dy / length
                                    ny = dx / length
                                    thickness = 4.0

                                    bridge_rows = [
                                        best_p1.y() + ny * thickness,
                                        best_p2.y() + ny * thickness,
                                        best_p2.y() - ny * thickness,
                                        best_p1.y() - ny * thickness,
                                    ]
                                    bridge_cols = [
                                        best_p1.x() + nx * thickness,
                                        best_p2.x() + nx * thickness,
                                        best_p2.x() - nx * thickness,
                                        best_p1.x() - nx * thickness,
                                    ]

                                    bridge_rows = [
                                        max(0, min(self.image.height() - 1, r))
                                        for r in bridge_rows
                                    ]
                                    bridge_cols = [
                                        max(0, min(self.image.width() - 1, c))
                                        for c in bridge_cols
                                    ]
                                    rr, cc = draw_polygon(
                                        bridge_rows, bridge_cols, mask.shape
                                    )
                                    mask[rr, cc] = 1

                            contours = find_contours(mask, 0.5)
                            if contours:
                                from scipy.interpolate import splev, splprep

                                contour = max(contours, key=len)
                                y, x = contour[:, 0], contour[:, 1]  # ty:ignore[not-subscriptable]

                                if len(x) > 4:
                                    if not np.allclose([x[0], y[0]], [x[-1], y[-1]]):
                                        x = np.append(x, x[0])
                                        y = np.append(y, y[0])
                                    try:
                                        tck, u = splprep([x, y], s=3.0, per=True)
                                        u_new = np.linspace(u.min(), u.max(), len(x))
                                        x_new, y_new = splev(u_new, tck)
                                        new_obj = [
                                            QPointF(nx, ny)
                                            for nx, ny in zip(x_new, y_new)
                                        ]
                                    except Exception:
                                        new_obj = [
                                            QPointF(nx, ny) for nx, ny in zip(x, y)
                                        ]
                                else:
                                    new_obj = [QPointF(nx, ny) for nx, ny in zip(x, y)]

                                self.objects[self.join_base_idx] = new_obj
                                self.objects.pop(clicked_idx)

                                if clicked_idx < self.join_base_idx:
                                    self.join_base_idx -= 1

                                self.objects_updated.emit()

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

    def _reset_interaction_state(self):
        self.join_base_idx = -1
        self.current_points = []
        self.drawing = False

    def _change_tool(self, new_tool: Tool):
        self.tool = new_tool
        self._reset_interaction_state()
        self.update()

    def set_tool_hand(self):
        self._change_tool(Tool.HAND)

    def set_tool_pen(self):
        self._change_tool(Tool.PEN)

    def set_tool_eraser(self):
        self._change_tool(Tool.ERASER)

    def set_tool_join(self):
        self._change_tool(Tool.JOIN)

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

        save_path = Path(Path(self.image_path).name).with_suffix(".npy")
        np.save(save_path, labels)
        print(f"Mask saved pure to {save_path}")
