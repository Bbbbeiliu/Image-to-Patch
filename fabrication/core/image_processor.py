import cv2
import numpy as np
import traceback
import math
from pathlib import Path
from typing import List, Tuple, Any, Optional
from PyQt5.QtCore import QPointF
from .file_utils import FileUtils
try:
    from geomdl import NURBS
    from geomdl import knotvector
    GEOMDL_AVAILABLE = True
except ImportError:
    GEOMDL_AVAILABLE = False
    print('NURBS fitting requires geomdl. Install with: pip install geomdl')

class AdvancedImageProcessor:

    @staticmethod
    def load_and_process_image(image_path: str, kernel_size: int=3) -> List[Tuple[np.ndarray, str]]:
        try:
            image_name = Path(image_path).name
            contours = []
            with open(image_path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f'Cannot read image: {image_path}')
                return contours
            if len(np.unique(image)) > 2:
                _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            else:
                binary = image
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            contours_data, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours_data:
                area = cv2.contourArea(contour)
                if area < 100:
                    continue
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) >= 3:
                    points = approx.squeeze()
                    if points.ndim == 1:
                        points = points.reshape(-1, 2)
                    if not np.array_equal(points[0], points[-1]):
                        points = np.vstack([points, points[0]])
                    contours.append((points, image_name))
            return contours
        except Exception as e:
            print(f'Image processing failed for {image_path}: {str(e)}')
            traceback.print_exc()
            return []

    @staticmethod
    def simplify_contour(points: np.ndarray, tolerance: float=1.0) -> np.ndarray:
        if len(points) < 3:
            return points

        def rdp_recursive(point_list, start_idx, end_idx):
            if end_idx <= start_idx + 1:
                return []
            max_dist = 0
            max_idx = start_idx
            start_pt = point_list[start_idx]
            end_pt = point_list[end_idx]
            for i in range(start_idx + 1, end_idx):
                pt = point_list[i]
                if np.array_equal(start_pt, end_pt):
                    dist = np.linalg.norm(pt - start_pt)
                else:
                    line_len = np.linalg.norm(end_pt - start_pt)
                    if line_len == 0:
                        dist = np.linalg.norm(pt - start_pt)
                    else:
                        t = max(0, min(1, np.dot(pt - start_pt, end_pt - start_pt) / (line_len * line_len)))
                        projection = start_pt + t * (end_pt - start_pt)
                        dist = np.linalg.norm(pt - projection)
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i
            result = []
            if max_dist > tolerance:
                rec1 = rdp_recursive(point_list, start_idx, max_idx)
                rec2 = rdp_recursive(point_list, max_idx, end_idx)
                result = rec1 + [point_list[max_idx]] + rec2
            return result
        simplified = [points[0]] + rdp_recursive(points, 0, len(points) - 1) + [points[-1]]
        return np.array(simplified)

    @staticmethod
    def smooth_contour_with_nurbs(points: np.ndarray, precision: float=0.5, num_control_points: Optional[int]=None) -> Tuple[List[QPointF], Any]:
        if not GEOMDL_AVAILABLE:
            print('geomdl is unavailable; using Bezier fitting.')
            _, nurbs_points = AdvancedImageProcessor.smooth_contour_with_cubic_bezier(points, int(precision * 40 + 10))
            return (nurbs_points, None)
        if len(points) < 3:
            return ([], None)
        points = points.squeeze()
        if points.ndim != 2 or points.shape[1] != 2:
            return ([], None)
        if not np.array_equal(points[0], points[-1]):
            points = np.vstack([points, points[0]])
        try:
            simplify_tolerance = max(0.5, min(5.0, (1.0 - precision) * 10))
            degree = 3 if precision > 0.3 else 2
            simplified_points = AdvancedImageProcessor.simplify_contour(points, simplify_tolerance)
            if len(simplified_points) < 3:
                simplified_points = points
            if num_control_points is not None:
                target_control_points = max(degree + 1, min(500, num_control_points))
            else:
                target_control_points = max(10, min(50, int(pow(precision, 1.5) * 40 + 10)))
            src_points = simplified_points
            src_len = len(src_points)
            if src_len <= target_control_points:
                selected_indices = list(range(src_len))
            else:
                step = src_len / target_control_points
                selected_indices = [int(i * step) for i in range(target_control_points)]
            control_points_list = []
            for idx in selected_indices:
                pt = src_points[idx]
                if len(pt) == 2:
                    control_points_list.append([float(pt[0]), float(pt[1]), 1.0])
                else:
                    control_points_list.append([float(pt[0]), float(pt[1]), float(pt[2]) if len(pt) > 2 else 1.0])
            while len(control_points_list) < degree + 1:
                pt = simplified_points[-1]
                control_points_list.append([float(pt[0]), float(pt[1]), 1.0])
            curve = NURBS.Curve()
            curve.degree = degree
            curve.ctrlpts = control_points_list
            curve.knotvector = knotvector.generate(curve.degree, len(curve.ctrlpts))
            sample_points = max(100, min(300, int(precision * 200 + 100)))
            curve.sample_size = sample_points
            curve_points = curve.evalpts
            nurbs_points = [QPointF(float(p[0]), float(p[1])) for p in curve_points]
            if len(nurbs_points) > 1 and nurbs_points[0] != nurbs_points[-1]:
                nurbs_points.append(nurbs_points[0])
            return (nurbs_points, curve)
        except Exception as e:
            print(f'NURBS fitting error: {str(e)}')
            traceback.print_exc()
            _, nurbs_points = AdvancedImageProcessor.smooth_contour_with_cubic_bezier(points, int(precision * 40 + 10))
            return (nurbs_points, None)

    @staticmethod
    def smooth_contour_with_cubic_bezier(points: np.ndarray, num_control_points: int=20) -> Tuple[List[QPointF], List[QPointF]]:
        if len(points) < 3:
            return ([], [])
        points = points.squeeze()
        if points.ndim != 2 or points.shape[1] != 2:
            return ([], [])
        if not np.array_equal(points[0], points[-1]):
            points = np.vstack([points, points[0]])
        n_points = len(points)
        if n_points <= 1:
            return ([], [])
        total_length = 0
        segment_lengths = []
        for i in range(n_points - 1):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            length = math.sqrt(dx * dx + dy * dy)
            segment_lengths.append(length)
            total_length += length
        if total_length == 0:
            return ([], [])
        control_points = []
        num_control_points = min(num_control_points, n_points)
        target_spacing = total_length / num_control_points
        current_length = 0
        next_target = target_spacing
        control_points.append(QPointF(float(points[0][0]), float(points[0][1])))
        for i in range(n_points - 1):
            seg_len = segment_lengths[i]
            while current_length + seg_len >= next_target and len(control_points) < num_control_points:
                t = (next_target - current_length) / seg_len
                x = points[i][0] + t * (points[i + 1][0] - points[i][0])
                y = points[i][1] + t * (points[i + 1][1] - points[i][1])
                control_points.append(QPointF(float(x), float(y)))
                next_target += target_spacing
            current_length += seg_len
        if len(control_points) > 1 and control_points[0] != control_points[-1]:
            control_points.append(control_points[0])
        if len(control_points) >= 4:
            bezier_points = []
            n = len(control_points) - 1
            for i in range(n):
                p0 = control_points[i - 1] if i > 0 else control_points[n - 1]
                p1 = control_points[i]
                p2 = control_points[i + 1]
                p3 = control_points[i + 2] if i < n - 1 else control_points[1]
                b0 = p1
                b1 = p1 + (p2 - p0) / 6.0
                b2 = p2 - (p3 - p1) / 6.0
                b3 = p2
                for t in np.linspace(0, 1, 30):
                    u = 1 - t
                    x = u ** 3 * b0.x() + 3 * u ** 2 * t * b1.x() + 3 * u * t ** 2 * b2.x() + t ** 3 * b3.x()
                    y = u ** 3 * b0.y() + 3 * u ** 2 * t * b1.y() + 3 * u * t ** 2 * b2.y() + t ** 3 * b3.y()
                    bezier_points.append(QPointF(x, y))
            return (control_points, bezier_points)
        return ([], [])
