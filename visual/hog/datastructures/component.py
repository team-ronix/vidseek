import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from visual.hog.datastructures.bbox_regressor import BBoxRegressor


class Component:
    def __init__(
        self,
        component_id,
        class_name,
        cell_w,
        cell_h,
        cell_size: int = 8,
        c_svm: float = 0.01,
        max_itr_svm: int = 10_000,
        alpha: float = 1000.0,
    ):
        self.id = component_id
        self.cls_name = class_name
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.cell_size = cell_size
        self.pixel_w = cell_w * cell_size
        self.pixel_h = cell_h * cell_size
        self.svm = LinearSVC(
            C=c_svm,
            max_iter=max_itr_svm,
            dual="auto",
            class_weight=None,
        )
        self.cal = None
        self.bbox_reg = BBoxRegressor(alpha=alpha)
        self.X_pos = np.array([])
        self.X_bg = np.array([])
        self.X_pos_other_classes = np.array([])
        self.X_cal = np.array([])
        self.y_cal = np.array([])
        self.bbr_X = np.array([])
        self.bbr_y = np.array([])

    def __len__(self):
        return self.X_pos.shape[0] + self.X_bg.shape[0] + self.X_pos_other_classes.shape[0]

    def counts(self) -> tuple[int, int, int]:
        return self.X_pos.shape[0], self.X_bg.shape[0], self.X_pos_other_classes.shape[0]

    
    def _check_svm_perf(self, X, y):
        y_pred = self.svm.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        print(f"Component {self.id} ({self.cls_name}) - SVM performance on training set: ")
        print(f"\taccuracy = {acc:.4f}, precision = {prec:.4f}, recall = {rec:.4f}, F1 = {f1:.4f}")

    def fit_svm(self, split_ratio: float | None = None) -> None:
        n_pos = self.X_pos.shape[0]
        n_neg = self.X_bg.shape[0] + self.X_pos_other_classes.shape[0]
        if n_pos == 0:
            print(f"Component {self.id} ({self.cls_name}): no positive samples - skipping.")
            return
        if n_neg == 0:
            print(f"Component {self.id} ({self.cls_name}): no negative samples - skipping.")
            return
        X = np.array([])
        if self.X_pos.shape[0] > 0:
            X = np.vstack([X, self.X_pos]) if X.size > 0 else self.X_pos
        if self.X_bg.shape[0] > 0:
            X = np.vstack([X, self.X_bg]) if X.size > 0 else self.X_bg
        if self.X_pos_other_classes.shape[0] > 0:
            X = np.vstack([X, self.X_pos_other_classes]) if X.size > 0 else self.X_pos_other_classes
        X = X.astype(np.float32)
        y = np.concatenate([np.ones(n_pos, dtype=int), -np.ones(n_neg, dtype=int)])
        if split_ratio is None:
            self.svm.fit(X, y)
            self._check_svm_perf(X, y)
            del X, y
            return
        if n_pos < 2 or n_neg < 2:
            print(
                f"Component {self.id} ({self.cls_name}): too few samples for a calibration split "
                f"({n_pos} pos, {n_neg} neg) - training on full set, skipping calibration."
            )
            self.svm.fit(X, y)
            self._check_svm_perf(X, y)
            del X, y
            return
        use_stratify = (n_pos >= 2) and (n_neg >= 2)
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=split_ratio,
                stratify=y if use_stratify else None,
                random_state=42,
            )
            del X, y
        except ValueError:
            print(
                f"Component {self.id} ({self.cls_name}): too few samples to split "
                f"({n_pos} pos, {n_neg} neg) - training on full set, skipping calibration."
            )
            self.svm.fit(X, y)
            self._check_svm_perf(X, y)
            del X, y
            return
        val_classes = set(y_val.tolist())
        for missing_class in ({1, -1} - val_classes):
            candidate_idxs = np.where(y_train == missing_class)[0]
            if len(candidate_idxs) == 0:
                break
            move_idx = candidate_idxs[0]
            X_val = np.vstack([X_val, X_train[move_idx : move_idx + 1]])
            y_val = np.append(y_val, y_train[move_idx])
            if len(candidate_idxs) > 1:
                X_train = np.delete(X_train, move_idx, axis=0)
                y_train = np.delete(y_train, move_idx)
        print(len(X_train), "training samples,", len(X_val), "calibration samples")
        self.svm.fit(X_train, y_train)
        self._check_svm_perf(X_train, y_train)
        self.X_cal = X_val
        self.y_cal = y_val
        del X_train, y_train
        

    def clear_easy_negatives(self, decision_threshold: float = -1.0) -> None:
        if self.X_bg.shape[0] == 0:
            return
        c1 = self.X_bg.shape[0]
        scrs = self.svm.decision_function(self.X_bg.astype(np.float32))
        mask = scrs > decision_threshold
        self.X_bg = self.X_bg[mask]
        c2 = self.X_bg.shape[0]
        print(f"Component {self.id} ({self.cls_name}): cleared {c1 - c2} easy negatives, {c2} hard negatives remain.")


    def fit_calibration(self) -> None:
        X_val = np.array([])
        y_val = np.array([])
        if self.X_cal.shape[0] > 0:
            X_val = self.X_cal
        if self.y_cal.shape[0] > 0:
            y_val = self.y_cal
        if X_val is None or y_val is None or len(X_val) == 0:
            print(
                f"Component {self.id} ({self.cls_name}): "
                f"no calibration samples - skipping calibration."
            )
            return
        found_cls = set(y_val.tolist())
        if len(found_cls) < 2:
            print(
                f"Component {self.id} ({self.cls_name}): "
                f"calibration set contains only class {found_cls} "
                f"({len(X_val)} samples) - skipping calibration."
            )
            return
        min_per_cls = min(int((y_val == c).sum()) for c in found_cls)
        if min_per_cls < 3:
            print(
                f"Component {self.id} ({self.cls_name}): "
                f"calibration set has only {min_per_cls} samples in the "
                f"minority class - skipping calibration (need >= 3 per class)."
            )
            return
        self.cal = CalibratedClassifierCV(self.svm, method="sigmoid", cv="prefit")
        self.cal.fit(X_val, y_val)
        
        
    def del_training_data(self) -> None:
        self.X_pos = np.array([])
        self.X_bg = np.array([])
        self.X_pos_other_classes = np.array([])
        self.X_cal = np.array([])
        self.y_cal = np.array([])
        self.bbr_X = np.array([])
        self.bbr_y = np.array([])
        
    def save_training_data(self, path):
        prfix = f"component_{self.cls_name}_{self.id}"
        np.save(path / f"{prfix}_X_pos.npy", self.X_pos)
        np.save(path / f"{prfix}_X_bg.npy", self.X_bg)
        np.save(path / f"{prfix}_X_pos_other_classes.npy", self.X_pos_other_classes)
        np.save(path / f"{prfix}_X_cal.npy", self.X_cal)
        np.save(path / f"{prfix}_y_cal.npy", self.y_cal)
        np.save(path / f"{prfix}_X_bbr.npy", self.bbr_X)
        np.save(path / f"{prfix}_y_bbr.npy", self.bbr_y)

    def load_training_data(self, path):
        prfix = f"component_{self.cls_name}_{self.id}"
        self.X_pos = np.load(path / f"{prfix}_X_pos.npy", mmap_mode="r")
        self.X_bg = np.load(path / f"{prfix}_X_bg.npy", mmap_mode="r")
        self.X_pos_other_classes = np.load(path / f"{prfix}_X_pos_other_classes.npy", mmap_mode="r")
        self.X_cal = np.load(path / f"{prfix}_X_cal.npy", mmap_mode="r")
        self.y_cal = np.load(path / f"{prfix}_y_cal.npy", mmap_mode="r")
        self.bbr_X = np.load(path / f"{prfix}_X_bbr.npy", mmap_mode="r")
        self.bbr_y = np.load(path / f"{prfix}_y_bbr.npy", mmap_mode="r")