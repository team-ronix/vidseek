import numpy as np
import cv2
from skimage.feature import hog
from visual.vrd_ml.vrd_dataset import BBox

CROP_SIZE = (32, 32)
HOG_PIXELS = 16
HOG_CELLS = 2
HOG_ORIENT = 9

_dummy = np.zeros((*CROP_SIZE, 3), dtype=np.uint8)
_hog_feat = hog(
    _dummy, orientations=HOG_ORIENT,
    pixels_per_cell=(HOG_PIXELS, HOG_PIXELS),
    cells_per_block=(HOG_CELLS, HOG_CELLS),
    channel_axis=-1, feature_vector=True,
)
HOG_DIM = _hog_feat.shape[0]
VISUAL_DIM = HOG_DIM * 3       # subject + object + union


def _safe_crop(
    image: np.ndarray,
    bbox: BBox,
    pad: int = 2,
) -> np.ndarray:
    H, W = image.shape[:2]
    x1 = max(0, int(bbox.x1) - pad)
    y1 = max(0, int(bbox.y1) - pad)
    x2 = min(W, int(bbox.x2) + pad)
    y2 = min(H, int(bbox.y2) + pad)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((*CROP_SIZE, 3), dtype=np.uint8)
    crop = image[y1:y2, x1:x2]
    return cv2.resize(crop, CROP_SIZE, interpolation=cv2.INTER_LINEAR)


def _hog_features(crop: np.ndarray) -> np.ndarray:
    feat = hog(
        crop,
        orientations=HOG_ORIENT,
        pixels_per_cell=(HOG_PIXELS, HOG_PIXELS),
        cells_per_block=(HOG_CELLS, HOG_CELLS),
        channel_axis=-1,
        feature_vector=True,
    )
    return feat.astype(np.float32)

class VisualFeatureExtractor:
    dim: int = VISUAL_DIM

    def extract(
        self,
        image: np.ndarray,
        subj_box: BBox,
        obj_box: BBox,
    ) -> np.ndarray:
        union_box = subj_box.union(obj_box)
        subj_crop = _safe_crop(image, subj_box)
        obj_crop = _safe_crop(image, obj_box)
        union_crop = _safe_crop(image, union_box)
        feat = np.concatenate([
            _hog_features(subj_crop),
            _hog_features(obj_crop),
            _hog_features(union_crop),
        ]).astype(np.float32)
        assert feat.shape == (VISUAL_DIM,), f"Expected {VISUAL_DIM}-d, got {feat.shape}"
        return feat

    def extract_batch(
        self,
        image: np.ndarray,
        pairs: list,  # list of (BBox, BBox)
    ) -> np.ndarray:
        return np.stack([self.extract(image, s, o) for s, o in pairs], axis=0)
