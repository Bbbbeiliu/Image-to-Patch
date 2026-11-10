import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

class YOLOMarkerDetector:

    def __init__(self, model_path: str, conf_threshold: float=0.5, nms_threshold: float=0.5, input_size: int=640, ball_diameter_mm: float=10.0):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.ball_diameter_mm = ball_diameter_mm
        self.net = None
        self._load_model()

    def _load_model(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f'Model file does not exist: {self.model_path}')
        print(f'[YOLO] Loading model: {self.model_path}')
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        print('[YOLO] Model loaded')

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, int, int, int, int], Tuple[int, int]]:
        original_h, original_w = image.shape[:2]
        scale = min(self.input_size / original_w, self.input_size / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_top = (self.input_size - new_h) // 2
        pad_left = (self.input_size - new_w) // 2
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        norm = rgb.astype(np.float32) / 255.0
        chw = norm.transpose(2, 0, 1)
        blob = np.expand_dims(chw, axis=0)
        return (blob, (scale, pad_left, pad_top, new_w, new_h), (original_h, original_w))

    def _postprocess(self, outputs: np.ndarray, preprocess_info: Tuple, original_shape: Tuple) -> List:
        scale, pad_left, pad_top, new_w, new_h = preprocess_info
        original_h, original_w = original_shape
        predictions = outputs[0].transpose(1, 0)
        bbox_data = predictions[:, :4]
        scores = predictions[:, 4]
        keep = scores > self.conf_threshold
        if not np.any(keep):
            return []
        bbox_data = bbox_data[keep]
        scores = scores[keep]
        cx, cy, w, h = (bbox_data[:, 0], bbox_data[:, 1], bbox_data[:, 2], bbox_data[:, 3])
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_640 = np.column_stack([x1, y1, x2, y2])
        boxes_no_pad = boxes_640.copy()
        boxes_no_pad[:, [0, 2]] -= pad_left
        boxes_no_pad[:, [1, 3]] -= pad_top
        boxes_original = boxes_no_pad / scale
        boxes_original[:, [0, 2]] = np.clip(boxes_original[:, [0, 2]], 0, original_w)
        boxes_original[:, [1, 3]] = np.clip(boxes_original[:, [1, 3]], 0, original_h)
        indices = cv2.dnn.NMSBoxes(boxes_original.tolist(), scores.tolist(), score_threshold=self.conf_threshold, nms_threshold=self.nms_threshold)
        detections = []
        if len(indices) > 0:
            indices = indices.flatten()
            for idx in indices:
                x1, y1, x2, y2 = boxes_original[idx]
                conf = scores[idx]
                detections.append([int(x1), int(y1), int(x2), int(y2), float(conf)])
        if detections:
            detections.sort(key=lambda d: (d[2] - d[0]) * (d[3] - d[1]), reverse=True)
        return detections

    def get_detection_info(self, image_path: str) -> Tuple[Optional[List], Optional[float]]:
        try:
            with open(image_path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f'[YOLO] Cannot read image: {image_path}, error: {e}')
            return (None, None)
        if img is None:
            print(f'[YOLO] Cannot decode image: {image_path}')
            return (None, None)
        blob, pre_info, orig_shape = self._preprocess(img)
        self.net.setInput(blob)
        outputs = self.net.forward()
        detections = self._postprocess(outputs, pre_info, orig_shape)
        if not detections:
            print(f'[YOLO] No marker detected: {image_path}')
            return (None, None)
        best = detections[0]
        x1, y1, x2, y2, conf = best
        width = x2 - x1
        height = y2 - y1
        pixel_diameter = (width + height) / 2.0
        if pixel_diameter <= 0:
            return (None, None)
        scale_mm_per_px = self.ball_diameter_mm / pixel_diameter
        print(f'[YOLO] Detected marker: confidence={conf:.3f}, pixel diameter={pixel_diameter:.1f}px, scale={scale_mm_per_px:.6f} mm/px')
        return (detections, scale_mm_per_px)

    def get_scale_from_image(self, image_path: str) -> Optional[float]:
        _, scale = self.get_detection_info(image_path)
        return scale
