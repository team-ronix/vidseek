from pathlib import Path
import numpy as np
import cv2


def load_image(src) -> np.ndarray | None:
    if isinstance(src, (str, Path)):
        return cv2.imread(str(src))
    return src


def calculate_iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(area_a + area_b - inter_area + 1e-6)