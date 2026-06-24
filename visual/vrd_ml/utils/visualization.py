import numpy as np
import cv2
import os
from typing import List, Tuple, Optional, Dict

from visual.vrd_ml.vrd_dataset import BBox, DetectedObject, RelationshipTriplet

# Colour palette (BGR)
COLORS = [
    (220,  80,  80), ( 80, 180,  80), ( 80, 120, 220),
    (220, 160,  40), (160,  80, 220), ( 40, 200, 200),
    (220,  80, 160), (140, 200,  80), ( 80, 160, 160),
]


def _color(idx: int) -> Tuple[int, int, int]:
    return COLORS[idx % len(COLORS)]


def draw_objects(
    image:   np.ndarray,
    objects: List[DetectedObject],
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and class labels on a copy of `image`."""
    out = image.copy()
    for i, obj in enumerate(objects):
        c   = _color(i)
        b   = obj.bbox
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, thickness)
        label = f"{obj.label} {obj.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), c, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_triplets(
    image:      np.ndarray,
    triplets:   List[RelationshipTriplet],
    max_draw:   int = 5,
    thickness:  int = 2,
) -> np.ndarray:
    """
    Draw the top `max_draw` relationship triplets on a copy of `image`.

    Each triplet is shown as:
      - subject box (solid)
      - object  box (dashed)
      - arrow from subject centre to object centre
      - predicate label at the midpoint of the arrow
    """
    out = image.copy()
    for k, trip in enumerate(triplets[:max_draw]):
        c    = _color(k)
        sb   = trip.subject.bbox
        ob   = trip.object_.bbox
        s_pt = (int(sb.center[0]), int(sb.center[1]))
        o_pt = (int(ob.center[0]), int(ob.center[1]))

        # Subject box
        cv2.rectangle(out, (int(sb.x1), int(sb.y1)), (int(sb.x2), int(sb.y2)), c, thickness)
        # Object box (dashed via segment drawing)
        _dashed_rect(out, ob, c, thickness)

        # Arrow
        cv2.arrowedLine(out, s_pt, o_pt, c, 1, tipLength=0.03, line_type=cv2.LINE_AA)

        # Predicate label at midpoint
        mx = (s_pt[0] + o_pt[0]) // 2
        my = (s_pt[1] + o_pt[1]) // 2
        label = trip.predicate
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(out, (mx - 2, my - th - 4), (mx + tw + 2, my + 2), (30, 30, 30), -1)
        cv2.putText(out, label, (mx, my - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)

        # Subject / object labels
        cv2.putText(out, trip.subject.label, (int(sb.x1), int(sb.y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
        cv2.putText(out, trip.object_.label, (int(ob.x1), int(ob.y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)

    return out


def _dashed_rect(img: np.ndarray, bbox: BBox, color: Tuple, thickness: int, gap: int = 8):
    """Draw a dashed rectangle (approximation)."""
    pts = [
        ((int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y1))),
        ((int(bbox.x2), int(bbox.y1)), (int(bbox.x2), int(bbox.y2))),
        ((int(bbox.x2), int(bbox.y2)), (int(bbox.x1), int(bbox.y2))),
        ((int(bbox.x1), int(bbox.y2)), (int(bbox.x1), int(bbox.y1))),
    ]
    for (p1, p2) in pts:
        _dashed_line(img, p1, p2, color, thickness, gap)


def _dashed_line(img, p1, p2, color, thickness, gap):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist   = max(1, int(np.hypot(dx, dy)))
    steps  = dist // (gap * 2)
    for i in range(steps):
        t0 = (i * 2 * gap) / dist
        t1 = min(1.0, (i * 2 * gap + gap) / dist)
        a  = (int(p1[0] + dx * t0), int(p1[1] + dy * t0))
        b  = (int(p1[0] + dx * t1), int(p1[1] + dy * t1))
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def save_result_grid(
    images:    List[np.ndarray],
    titles:    List[str],
    save_path: str,
    cols:      int = 2,
    cell_w:    int = 400,
    cell_h:    int = 300,
):
    """Save a grid of images with titles to `save_path`."""
    n    = len(images)
    rows = (n + cols - 1) // cols
    grid = np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) * 240

    for k, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(k, cols)
        resized = cv2.resize(img, (cell_w, cell_h - 24))
        y0 = r * cell_h
        x0 = c * cell_w
        grid[y0 + 24: y0 + cell_h, x0: x0 + cell_w] = resized
        cv2.putText(grid, title, (x0 + 4, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, grid)
    print(f"[viz] Saved grid -> {save_path}")
