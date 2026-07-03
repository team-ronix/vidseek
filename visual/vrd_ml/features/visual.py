import numpy as np
import cv2
from skimage.feature import hog
from visual.vrd_ml.vrd_dataset import BBox
from OCR.utils.Hog import HoG, calc_gradients
from PIL import Image

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
# one for subject and one for object and one for union region
VISUAL_DIM = HOG_DIM * 3


def _safe_crop(image, bbox, pad=2):
    H, W = image.shape[:2]
    x1 = max(0, int(bbox.x1) - pad)
    y1 = max(0, int(bbox.y1) - pad)
    x2 = min(W, int(bbox.x2) + pad)
    y2 = min(H, int(bbox.y2) + pad)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((*CROP_SIZE, 3), dtype=np.uint8)
    crop = image[y1:y2, x1:x2]
    return cv2.resize(crop, CROP_SIZE, interpolation=cv2.INTER_LINEAR)


def _hog_features(crop):
    img = Image.fromarray(crop).convert("L")
    mag, orient = calc_gradients(np.array(img))
    feat = HoG(orient, mag, cell_size=HOG_PIXELS, num_bins=HOG_ORIENT, block_size=HOG_CELLS)
    return feat.astype(np.float32)


class VisualFeatureExtractor:
    dim = VISUAL_DIM

    def extract(self, image, subj_box, obj_box):
        union_box = subj_box.union(obj_box)
        subj_crop = _safe_crop(image, subj_box)
        obj_crop = _safe_crop(image, obj_box)
        union_crop = _safe_crop(image, union_box)
        feat = np.concatenate([
            _hog_features(subj_crop),
            _hog_features(obj_crop),
            _hog_features(union_crop),
        ]).astype(np.float32)
        return feat

    def extract_batch(self, image, pairs):
        return np.stack([self.extract(image, s, o) for s, o in pairs], axis=0)
