import numpy as np
from PyQt5.QtCore import QPointF, QRectF
from PyQt5.QtGui import QColor, QBrush, QPen
import random
import math

class Contour:

    def __init__(self, points: np.ndarray, contour_id: int, source_image: str='', label: int=0):
        self.id = contour_id
        self.original_points = points
        self.nurbs_points = []
        self.position = QPointF(0, 0)
        self.scale = 1.0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.is_selected = False
        self.source_image = source_image
        self.bounding_box = QRectF()
        self.color = self.generate_color(label if label > 0 else contour_id)
        self.label = label
        self.label_text = str(label) if label > 0 else ''
        self.label_font_size = 12
        self.actual_width_cm = 0.0
        self.actual_height_cm = 0.0
        self.pixel_scale_mm_per_px = None
        self.nurbs_curve = None
        self.precision = 0.5
        self.control_points = 50
        self.calculate_bounding_box()

    def calculate_bounding_box(self):
        if len(self.original_points) == 0:
            return
        points = self.original_points.squeeze()
        if points.ndim != 2:
            return
        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])
        self.bounding_box = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def generate_color(self, idx):
        colors = [QColor(255, 0, 0), QColor(0, 180, 0), QColor(0, 100, 255), QColor(255, 180, 0), QColor(180, 0, 180), QColor(0, 180, 180), QColor(255, 100, 0), QColor(150, 0, 255)]
        return colors[idx % len(colors)]

    def get_display_rect(self):
        if self.bounding_box.isNull():
            return QRectF()
        width = self.bounding_box.width() * self.scale_x
        height = self.bounding_box.height() * self.scale_y
        center_x = self.position.x() + width / 2
        center_y = self.position.y() + height / 2
        return QRectF(center_x - width / 2, center_y - height / 2, width, height)

    def get_geometric_center(self) -> QPointF:
        if not self.nurbs_points:
            return QPointF(0, 0)
        sum_x = sum((p.x() for p in self.nurbs_points))
        sum_y = sum((p.y() for p in self.nurbs_points))
        n = len(self.nurbs_points)
        local_center = QPointF(sum_x / n, sum_y / n)
        display_rect = self.get_display_rect()
        if display_rect.isNull():
            return QPointF(0, 0)
        bbox_tl = self.bounding_box.topLeft()
        scale = self.scale
        display_x = display_rect.left() + (local_center.x() - bbox_tl.x()) * scale
        display_y = display_rect.top() + (local_center.y() - bbox_tl.y()) * scale
        return QPointF(display_x, display_y)
    from PyQt5.QtCore import QPointF

    def get_label_position(self, pixels_per_cm: float, font_size_mm: float, min_size_mm: float, step_ratio: float=0.5):
        if not self.nurbs_points or len(self.nurbs_points) < 3:
            return (None, 0)
        min_scale = min(self.scale_x, self.scale_y)
        font_size_local = font_size_mm * (pixels_per_cm / 10) / min_scale
        min_size_local = min_size_mm * (pixels_per_cm / 10) / min_scale
        bbox_width = self.bounding_box.width()
        bbox_height = self.bounding_box.height()
        if bbox_width < min_size_local or bbox_height < min_size_local:
            return (None, 0)
        pts = [(p.x(), p.y()) for p in self.nurbs_points]
        x_min = min((p[0] for p in pts))
        x_max = max((p[0] for p in pts))
        step = max(1.0, font_size_local * step_ratio)
        best_point = None
        best_score = -1
        best_vertical_dist = 0
        x = x_min
        while x <= x_max:
            intersections = []
            for j in range(len(pts)):
                p1 = pts[j]
                p2 = pts[(j + 1) % len(pts)]
                if p1[0] <= x <= p2[0] or p2[0] <= x <= p1[0]:
                    if p1[0] == p2[0]:
                        continue
                    t = (x - p1[0]) / (p2[0] - p1[0])
                    y = p1[1] + t * (p2[1] - p1[1])
                    intersections.append(y)
            if intersections:
                intersections.sort()
                for k in range(0, len(intersections), 2):
                    if k + 1 >= len(intersections):
                        break
                    y1 = intersections[k]
                    y2 = intersections[k + 1]
                    dist_y = y2 - y1
                    if dist_y >= font_size_local:
                        y_mid = (y1 + y2) / 2
                        h_intersections = []
                        for j in range(len(pts)):
                            p1 = pts[j]
                            p2 = pts[(j + 1) % len(pts)]
                            if p1[1] <= y_mid <= p2[1] or p2[1] <= y_mid <= p1[1]:
                                if p1[1] == p2[1]:
                                    continue
                                t = (y_mid - p1[1]) / (p2[1] - p1[1])
                                x_h = p1[0] + t * (p2[0] - p1[0])
                                h_intersections.append(x_h)
                        if h_intersections:
                            h_intersections.sort()
                            for hk in range(0, len(h_intersections), 2):
                                if hk + 1 >= len(h_intersections):
                                    break
                                x1_h = h_intersections[hk]
                                x2_h = h_intersections[hk + 1]
                                if x1_h <= x <= x2_h:
                                    dist_x = x2_h - x1_h
                                    if dist_x >= font_size_local:
                                        score = dist_y
                                        if score > best_score:
                                            best_score = score
                                            best_vertical_dist = dist_y
                                            display_rect = self.get_display_rect()
                                            bbox_tl = self.bounding_box.topLeft()
                                            display_x = display_rect.left() + (x - bbox_tl.x()) * self.scale_x
                                            display_y = display_rect.top() + (y_mid - bbox_tl.y()) * self.scale_y
                                            best_point = QPointF(display_x, display_y)
                                        break
            x += step
        if best_point is not None:
            return (best_point, best_vertical_dist)
        else:
            return (None, 0)

    def set_size(self, width_cm: float, height_cm: float, pixels_per_cm: float, pixel_scale_mm_per_px: float=None):
        if self.bounding_box.width() > 0 and self.bounding_box.height() > 0:
            width_px = width_cm * pixels_per_cm
            height_px = height_cm * pixels_per_cm
            self.scale_x = width_px / self.bounding_box.width()
            self.scale_y = height_px / self.bounding_box.height()
            self.scale = (self.scale_x + self.scale_y) / 2.0
            self.actual_width_cm = width_cm
            self.actual_height_cm = height_cm
            self.pixel_scale_mm_per_px = pixel_scale_mm_per_px

    def update_label_size(self, pixels_per_cm: float):
        display_rect = self.get_display_rect()
        if display_rect.isNull():
            self.label_font_size = 12
            return
        diameter_px = display_rect.width() / 5
        min_diameter_px = 2 * pixels_per_cm / 10
        max_diameter_px = 10 * pixels_per_cm / 10
        diameter_px = max(min_diameter_px, min(diameter_px, max_diameter_px))
        self.label_font_size = int(diameter_px * 0.6)
        min_font_size = int(min_diameter_px * 0.4)
        max_font_size = int(max_diameter_px * 0.6)
        self.label_font_size = max(min_font_size, min(self.label_font_size, max_font_size))
