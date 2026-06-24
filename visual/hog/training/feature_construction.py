import gc
import numpy as np
import cv2
from visual.hog.utils import load_image


def construct_Xpos_Xneg(
    detector,
    pos_patches: dict[str, list[np.ndarray]],
    comp_labels: dict[str, np.ndarray],
    neg_images: list[tuple[str, list]],
    neg_patches_per_image: int,
) -> None:
    n_other_classes = len(detector.classes) - 1
    per_class_other_ratio = detector.other_classes_total_ratio / max(1, n_other_classes)
    for cls in detector.classes:
        pos_patches_cls = pos_patches[cls]
        comp_labels_cls = comp_labels[cls]
        for comp in detector.cls_comps[cls]:
            comp_w, comp_h = comp.pixel_w, comp.pixel_h

            # positives
            pos_patches_comp = [
                p for p, label in zip(pos_patches_cls, comp_labels_cls)
                if label == comp.id
            ]
            if not pos_patches_comp:
                continue
            comp.X_pos = np.array([
                detector._extract_custom_hog(cv2.resize(p, (comp_w, comp_h)))
                for p in pos_patches_comp
            ], dtype=np.float16)
            n_pos = len(comp.X_pos)

            # background negatives
            n_bg_target = int(n_pos * detector.bg_multiplier)
            bg_patches = _sample_background_patches_for_component(
                neg_images,
                patch_size=(comp_w, comp_h),
                n_patches_per_image=neg_patches_per_image,
                max_patches=n_bg_target,
            )
            comp.X_bg = np.array([detector._extract_custom_hog(p) for p in bg_patches], dtype=np.float16)
            del bg_patches

            # other-class positives (negatives from other classes)
            n_other_per_cls = max(1, int(n_pos * per_class_other_ratio))
            other_feats = []
            for other_cls in detector.classes:
                if other_cls == cls:
                    continue
                other_patches = pos_patches[other_cls]
                n_sample = min(len(other_patches), n_other_per_cls)
                if n_sample == 0:
                    continue
                indices = np.random.choice(len(other_patches), size=n_sample, replace=False)
                other_feats.extend([
                    detector._extract_custom_hog(
                        cv2.resize(other_patches[i], (comp_w, comp_h))
                    )
                    for i in indices
                ])
            comp.X_pos_other_classes = np.array(
                other_feats, dtype=np.float16
            ) if len(other_feats) > 0 else np.array([], dtype=np.float16)
            print(
                f"Class: {cls} | Comp: {comp.id}\n"
                f"\tPositives: {len(comp.X_pos)}\n"
                f"\tBackground: {len(comp.X_bg)}\n"
                f"\tOther-class: {len(comp.X_pos_other_classes)}\n"
            )
        del pos_patches_cls, comp_labels_cls
        gc.collect()



def _patch_overlaps_gt(
    x0: int, y0: int, x1: int, y1: int,
    gt_boxes: list,
    iou_thresh: float,
) -> bool:
    for gx0, gy0, gx1, gy1 in gt_boxes:
        ix0 = max(x0, gx0)
        iy0 = max(y0, gy0)
        ix1 = min(x1, gx1)
        iy1 = min(y1, gy1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        if inter == 0.0:
            continue
        area_patch = (x1 - x0) * (y1 - y0)
        area_gt = (gx1 - gx0) * (gy1 - gy0)
        iou = inter / (area_patch + area_gt - inter + 1e-6)
        if iou > iou_thresh:
            return True
    return False


def _sample_background_patches_for_component(
    neg_images: list[tuple[str, list]],
    patch_size: tuple[int, int],
    n_patches_per_image: int,
    max_patches: int,
    iou_thresh: float = 0.3,
) -> list[np.ndarray]:
    neg_patches: list[np.ndarray] = []
    pw, ph = patch_size
    indices = np.random.permutation(len(neg_images))
    for i, idx in enumerate(indices):
        if len(neg_patches) >= max_patches:
            break
        img_path, gt_boxes = neg_images[int(idx)]
        img = load_image(img_path)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        if iw < pw or ih < ph:
            del img
            continue
        cur = 0
        attempts, max_attempts = 0, 50 * n_patches_per_image
        while cur < n_patches_per_image and attempts < max_attempts:
            attempts += 1
            x0 = np.random.randint(0, iw - pw + 1)
            y0 = np.random.randint(0, ih - ph + 1)
            x1, y1 = x0 + pw, y0 + ph
            if gt_boxes and _patch_overlaps_gt(x0, y0, x1, y1, gt_boxes, iou_thresh):
                continue
            patch = img[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            neg_patches.append(patch.copy())
            cur += 1
            del patch
        del img
        if (i + 1) % 100 == 0:
            gc.collect()
    return neg_patches