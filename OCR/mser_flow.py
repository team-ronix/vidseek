from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import joblib
import numpy as np

try:
    from .ovo_svm import OvO_SVM
except ImportError:
    from ovo_svm import OvO_SVM


Box = Tuple[int, int, int, int]

IMG_SIZE = 128
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "from_scratch_SVM"
DEFAULT_LABEL_ENCODER_PATH = DEFAULT_MODEL_DIR / "OvO_SVM_label_encoder.joblib"


def merge_boxes(boxes: Sequence[Sequence[int]], threshold: float = 0.3) -> List[Box]:
    merged: List[Box] = []

    for box in boxes:
        x, y, w, h = map(int, box)
        if w == 0 or h == 0:
            continue

        added = False

        for index, (mx, my, mw, mh) in enumerate(merged):
            xi1 = max(x, mx)
            yi1 = max(y, my)
            xi2 = min(x + w, mx + mw)
            yi2 = min(y + h, my + mh)

            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            union = (w * h) + (mw * mh) - inter
            iou = inter / (union + 1e-6)

            if iou > threshold:
                nx = min(x, mx)
                ny = min(y, my)
                nw = max(x + w, mx + mw) - nx
                nh = max(y + h, my + mh) - ny
                merged[index] = (nx, ny, nw, nh)
                added = True
                break

        if not added:
            merged.append((x, y, w, h))

    return merged


def filter_char_boxes(boxes: Sequence[Sequence[int]], lower_bound: float = 0.1, higher_bound: float = 2.5) -> List[Box]:
    filtered: List[Box] = []

    for box in boxes:
        x, y, w, h = map(int, box)
        if w == 0 or h == 0:
            continue

        aspect_ratio = w / (h + 1e-6)
        if lower_bound <= aspect_ratio <= higher_bound:
            filtered.append((x, y, w, h))

    return filtered


def sort_boxes_reading_order(boxes: Sequence[Sequence[int]], y_thresh: int = 10) -> List[Box]:
    sorted_boxes = sorted((tuple(map(int, box)) for box in boxes), key=lambda box: box[1])

    lines: List[List[Box]] = []

    for box in sorted_boxes:
        x, y, w, h = box
        placed = False

        for line in lines:
            if abs(line[0][1] - y) < y_thresh:
                line.append(box)
                placed = True
                break

        if not placed:
            lines.append([box])

    for line in lines:
        line.sort(key=lambda item: item[0])

    return [box for line in lines for box in line]


def merge_char_words(boxes: Sequence[Sequence[int]], x_thresh: int = 20, y_thresh: int = 10) -> List[List[Box]]:
    boxes = [tuple(map(int, box)) for box in boxes]
    used = [False] * len(boxes)
    words: List[List[Box]] = []

    for index in range(len(boxes)):
        if used[index]:
            continue

        word = [boxes[index]]
        used[index] = True
        changed = True

        while changed:
            changed = False

            for other_index in range(len(boxes)):
                if used[other_index]:
                    continue

                x1, y1, w1, h1 = boxes[other_index]

                for wx, wy, ww, wh in word:
                    if abs(wy - y1) < y_thresh:
                        if abs((wx + ww) - x1) < x_thresh or abs((x1 + w1) - wx) < x_thresh:
                            word.append(boxes[other_index])
                            used[other_index] = True
                            changed = True
                            break

        words.append(sorted(word, key=lambda box: box[0]))

    return words


def get_word_boxes(words: Sequence[Sequence[Sequence[int]]]) -> List[Box]:
    word_boxes: List[Box] = []

    for word in words:
        x_min = min(x for x, y, w, h in word)
        y_min = min(y for x, y, w, h in word)
        x_max = max(x + w for x, y, w, h in word)
        y_max = max(y + h for x, y, w, h in word)
        word_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return word_boxes


def calc_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gx_kernel = np.array([[-1, 0, 1]], dtype=np.float32)
    gy_kernel = np.array([[-1], [0], [1]], dtype=np.float32)

    image = image.astype(np.float32)
    gx = cv2.filter2D(image, -1, gx_kernel)
    gy = cv2.filter2D(image, -1, gy_kernel)

    gradients_magnitude = np.sqrt(gx**2 + gy**2)
    gradients_orientation = (np.arctan2(gy, gx) * 180 / np.pi) % 180
    return gradients_magnitude, gradients_orientation


def hog(orientations: np.ndarray, magnitudes: np.ndarray, cell_size: int = 16, num_bins: int = 9, block_size: int = 2) -> np.ndarray:
    bin_size = 180 / num_bins
    height, width = orientations.shape
    cells_y = height // cell_size
    cells_x = width // cell_size

    histograms = np.zeros((cells_y, cells_x, num_bins), dtype=np.float32)

    for cell_y in range(cells_y):
        for cell_x in range(cells_x):
            y0 = cell_y * cell_size
            x0 = cell_x * cell_size

            for y in range(cell_size):
                for x in range(cell_size):
                    angle = orientations[y0 + y, x0 + x]
                    magnitude = magnitudes[y0 + y, x0 + x]
                    bin_idx = int(angle / bin_size) % num_bins
                    histograms[cell_y, cell_x, bin_idx] += magnitude

    features: List[float] = []
    for y in range(cells_y - 1):
        for x in range(cells_x - 1):
            block = histograms[y : y + block_size, x : x + block_size].flatten()
            block = block / (np.linalg.norm(block) + 1e-6)
            features.extend(block.tolist())

    return np.asarray(features, dtype=np.float32)


def load_model(model_dir: Path | str = DEFAULT_MODEL_DIR) -> OvO_SVM:
    return OvO_SVM.load(str(model_dir))


def load_label_encoder(label_encoder_path: Path | str = DEFAULT_LABEL_ENCODER_PATH):
    return joblib.load(str(label_encoder_path))


def predict_character(features: np.ndarray, model: OvO_SVM, label_encoder) -> Tuple[str, float, np.ndarray]:
    features = np.asarray(features)
    if features.ndim == 1:
        features = features.reshape(1, -1)

    predicted_label = model.predict(features)
    probabilities = model.predict_proba(features)
    confidence = model.confidence(features)
    decoded_label = label_encoder.inverse_transform(predicted_label)[0]
    return decoded_label, float(confidence[0]), probabilities


def prepare_character_image(gray: np.ndarray, box: Box) -> np.ndarray:
    x, y, w, h = box
    char_img = gray[y : y + h, x : x + w]
    padded = cv2.copyMakeBorder(char_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
    blurred = cv2.GaussianBlur(padded, (5, 5), 0)
    binarized = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )
    resized = cv2.resize(binarized, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    eroded = cv2.erode(resized, np.ones((5, 5), np.uint8), iterations=2)
    dilated = cv2.dilate(eroded, np.ones((3, 3), np.uint8), iterations=5)
    return dilated.astype(np.float32) / 255.0


def extract_text_from_image(
    image_path: Path | str,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    label_encoder_path: Path | str = DEFAULT_LABEL_ENCODER_PATH,
    show_debug: bool = False,
) -> str:
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    mser.setMinArea(300)

    _, boxes = mser.detectRegions(gray)
    unique_boxes = merge_boxes([list(box) for box in set(tuple(box) for box in boxes)], threshold=0.3)
    unique_boxes = filter_char_boxes(unique_boxes, lower_bound=0.25, higher_bound=1.3)
    unique_boxes = sort_boxes_reading_order(unique_boxes, y_thresh=10)
    words = merge_char_words(unique_boxes, x_thresh=20, y_thresh=30)

    model = load_model(model_dir)
    label_encoder = load_label_encoder(label_encoder_path)

    text_parts: List[str] = []
    for word in words:
        for box in word:
            normalized_char_img = prepare_character_image(gray, box)
            magnitudes, orientations = calc_gradients(normalized_char_img)
            features = hog(orientations, magnitudes)
            label, confidence, _ = predict_character(features, model, label_encoder)

            if show_debug:
                print(f"Predicted label: {label} (confidence: {confidence:.2f})")

            text_parts.append(label)
        text_parts.append(" ")

    return "".join(text_parts).strip()