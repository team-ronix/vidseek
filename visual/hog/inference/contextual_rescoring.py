import os
import pickle
import numpy as np
import heapq
from sklearn.svm import SVC
from tqdm import tqdm
from visual.hog.datastructures.voc_dataset import VOCDataset
from visual.hog.utils import calculate_iou


def _context_vector(scores: dict[str, float], classes: list[str]) -> np.ndarray:
    # return highest score for each class
    return np.array([scores.get(cls, 0.0) for cls in classes])


def _build_feature(
    score: float,
    box: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    context: np.ndarray,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    geo = np.array([
        score,
        float(x1) / img_w,
        float(y1) / img_h,
        float(x2) / img_w,
        float(y2) / img_h,
    ])
    return np.concatenate([geo, context])


class ContextualRescorer:
    def __init__(
        self,
        classes: list[str],
        iou_thresh: float = 0.5,
        C: float = 1.0,
        detection_threshold: float = 0.5,
        neg_size: int = 30_000,
    ):
        self.classes = classes
        self.iou_thresh = iou_thresh
        self.C = C
        self.detection_threshold = detection_threshold
        self.neg_size = neg_size
        self.rescorers: dict[str, SVC] = {}
        self.fitted: bool = False

    def _save_checkpoint(self, checkpoint_path, X_pos, X_neg, next_neg_order, last_idx):
        tmp_path = f"{checkpoint_path}.tmp"
        with open(tmp_path, 'wb') as f:
            pickle.dump({
                'X_pos': X_pos,
                'X_neg': X_neg,
                'next_neg_order': next_neg_order,
                'last_idx': last_idx,
            }, f)
        os.replace(tmp_path, checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
        return ckpt['X_pos'], ckpt['X_neg'], ckpt['next_neg_order'], ckpt['last_idx']

    def fit(
        self,
        detector,
        dataset: VOCDataset,
        max_images: int | None = None,
        checkpoint_path: str | None = None,
        checkpoint_every: int = 50,
    ) -> None:
        print("\nContextual Rescoring - collecting training features")
        n = len(dataset) if max_images is None else min(len(dataset), max_images)
        start_idx = 0
        X_pos: dict[str, list[np.ndarray]] = {cls: [] for cls in self.classes}
        X_neg: dict[str, list[tuple[float, int, np.ndarray]]] = {cls: [] for cls in self.classes}
        next_neg_order = 0

        if checkpoint_path and os.path.exists(checkpoint_path):
            X_pos, X_neg, next_neg_order, last_idx = self._load_checkpoint(checkpoint_path)
            for cls in self.classes:
                if cls not in X_pos:
                    X_pos[cls] = []
                if cls not in X_neg:
                    X_neg[cls] = []
            start_idx = last_idx + 1
            print(f"[Checkpoint] Resuming from image index {start_idx} "
                  f"(loaded from {checkpoint_path})")
        neg_order_val = next_neg_order

        def _next_neg_order() -> int:
            nonlocal neg_order_val
            v = neg_order_val
            neg_order_val += 1
            return v

        if start_idx >= n:
            print("[Checkpoint] All images already processed - skipping to SVM fitting.")
        else:
            processed_this_call = 0
            for idx in tqdm(range(start_idx, n), desc="Collecting Rescoring Features", initial=start_idx, total=n):
                img = dataset.get_image(idx)
                if img is None:
                    processed_this_call += 1
                    if checkpoint_path and processed_this_call % checkpoint_every == 0:
                        self._save_checkpoint(checkpoint_path, X_pos, X_neg, neg_order_val, idx)
                    continue
                ih, iw = img.shape[:2]
                gt_boxes, gt_labels = dataset.get_annotation(idx)
                detected_boxes, detected_scores, detected_labels = detector.detect(
                    img,
                    overlap_threshold=0.3,
                    threshold=self.detection_threshold,
                    use_context=False,
                )

                if detected_boxes:
                    best_per_class: dict[str, float] = {}
                    for box, score, label in zip(detected_boxes, detected_scores, detected_labels):
                        if label not in best_per_class or score > best_per_class[label]:
                            best_per_class[label] = score

                    context = _context_vector(best_per_class, self.classes)
                    for box, score, label in zip(detected_boxes, detected_scores, detected_labels):
                        g = _build_feature(score, box, iw, ih, context)
                        is_tp = any(
                            gt_label == label
                            and calculate_iou(box, gt_box) > self.iou_thresh
                            for gt_box, gt_label in zip(gt_boxes, gt_labels)
                        )
                        if is_tp:
                            X_pos[label].append(g)
                        else:
                            # keep the top-K highest-scoring FPs via a min-heap so
                            # the rescorer focuses on suppressing the most confident
                            # false positives. neg_order breaks score ties without
                            # comparing numpy arrays.
                            if len(X_neg[label]) < self.neg_size:
                                heapq.heappush(X_neg[label], (score, _next_neg_order(), g))
                            elif score > X_neg[label][0][0]:
                                heapq.heapreplace(X_neg[label], (score, _next_neg_order(), g))

                del img
                processed_this_call += 1
                if checkpoint_path and processed_this_call % checkpoint_every == 0:
                    self._save_checkpoint(
                        checkpoint_path, X_pos, X_neg, neg_order_val, idx
                    )
                    tqdm.write(f"[Checkpoint] Saved at idx={idx}")

        if checkpoint_path:
            last_done_idx = n - 1
            self._save_checkpoint(checkpoint_path, X_pos, X_neg, neg_order_val, last_done_idx)

        # Step 4: fit one SVM per class.
        print("Contextual Rescoring - fitting per-class SVMs")
        for cls in self.classes:
            if not X_pos[cls] and not X_neg[cls]:
                print(f"Warning: no training detections for '{cls}', skipping.")
                continue

            X_cls = np.array([g for g in X_pos[cls]]
                             + [g for _, _, g in X_neg[cls]])
            y_cls = np.array([1] * len(X_pos[cls])
                             + [-1] * len(X_neg[cls]))

            if len(set(y_cls)) < 2:
                print(f"Warning: only one class present for '{cls}' "
                      f"(all TP or all FP), skipping.")
                continue

            svm = SVC(
                kernel="poly",
                degree=2,
                C=self.C,
                probability=True,
                class_weight="balanced",
            )
            svm.fit(X_cls, y_cls)
            self.rescorers[cls] = svm
            n_tp = int((y_cls == 1).sum())
            print(
                f"[{cls}]: {len(X_cls)} detections  "
                f"({n_tp} TP / {len(X_cls) - n_tp} FP)"
            )
        self.fitted = True
        print("Contextual Rescoring - training complete.\n")
        if checkpoint_path and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"[Checkpoint] Training complete - removed {checkpoint_path}")

    def rescore(
        self,
        boxes: list,
        scores: list[float],
        labels: list[str],
        img_w: int,
        img_h: int,
    ) -> list[float]:
        if not self.fitted or not boxes:
            return scores
        best_per_class: dict[str, float] = {}
        for score, label in zip(scores, labels):
            if label not in best_per_class or score > best_per_class[label]:
                best_per_class[label] = score
        context = _context_vector(best_per_class, self.classes)
        new_scores = []
        for box, score, label in zip(boxes, scores, labels):
            if label not in self.rescorers:
                new_scores.append(score)
                continue
            g = _build_feature(score, box, img_w, img_h, context).reshape(1, -1)
            prob_tp = float(self.rescorers[label].predict_proba(g)[0, 1])
            new_scores.append(prob_tp)
        return new_scores