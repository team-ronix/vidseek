import numpy as np
from sklearn.linear_model import Ridge


# each component has its own regressor that learns to refine the raw sliding-window proposals toward ground-truth boxes
# that bounding box regressor consists of four independent Ridge regressors that map a component's HOG feature vector 
# to parametric offsets (tx, ty, tw, th) so that a raw sliding-window proposal can be shifted and resized toward the nearest ground-truth box.
class BBoxRegressor:
    def __init__(self, alpha: float = 1000.0):
        self.alpha = alpha
        self.regs = [Ridge(alpha=alpha) for _ in range(4)]
        self.fitted = False

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        # features: (N, D) array of HOG feature vectors for the proposals (proposal -> component window)
        # targets: (N, 4) array of regression targets (tx, ty, tw, th) for the proposals
        # tx = (gt_cx - p_cx) / p_w, ty = (gt_cy - p_cy) / p_h, tw = log(gt_w / p_w), th = log(gt_h / p_h)
        if len(features) == 0:
            return
        for k, reg in enumerate(self.regs):
            reg.fit(features, targets[:, k])
        self.fitted = True

    def predict(self, feat: np.ndarray) -> np.ndarray:
        if not self.fitted:
            if feat.ndim == 1:
                return np.zeros(4, dtype=np.float32)
            else:
                return np.zeros((feat.shape[0], 4), dtype=np.float32)
        if feat.ndim == 1:
            feat = feat.reshape(1, -1)
        # prediction order: tx, ty, tw, th
        # For each regressor, predict on all samples, then stack as columns to get (n_samples, 4)
        return np.column_stack([reg.predict(feat) for reg in self.regs]).astype(np.float32)

    @staticmethod
    def apply_deltas_batch(
        boxes: list,
        all_deltas: np.ndarray,
        clip_to: tuple[int, int] | None = None,
    ) -> list:
        n = len(boxes)
        if n == 0:
            return boxes
        b = np.array(boxes, dtype=np.float64)
        p_w = b[:, 2] - b[:, 0]
        p_h = b[:, 3] - b[:, 1]
        p_cx = b[:, 0] + 0.5 * p_w
        p_cy = b[:, 1] + 0.5 * p_h

        tx = all_deltas[:, 0]
        ty = all_deltas[:, 1]
        tw = np.clip(all_deltas[:, 2], -4.0, 4.0)
        th = np.clip(all_deltas[:, 3], -4.0, 4.0)

        gt_cx = tx * p_w + p_cx
        gt_cy = ty * p_h + p_cy
        gt_w = np.exp(tw) * p_w
        gt_h = np.exp(th) * p_h

        x0s = np.round(gt_cx - 0.5 * gt_w).astype(int)
        y0s = np.round(gt_cy - 0.5 * gt_h).astype(int)
        x1s = np.round(gt_cx + 0.5 * gt_w).astype(int)
        y1s = np.round(gt_cy + 0.5 * gt_h).astype(int)

        if clip_to is not None:
            img_w, img_h = clip_to
            x0s = np.clip(x0s, 0, img_w - 1)
            y0s = np.clip(y0s, 0, img_h - 1)
            x1s = np.clip(x1s, 0, img_w)
            y1s = np.clip(y1s, 0, img_h)

        valid = (x1s > x0s) & (y1s > y0s)
        result = []
        for i in range(n):
            if valid[i]:
                result.append((int(x0s[i]), int(y0s[i]), int(x1s[i]), int(y1s[i])))
            else:
                result.append(boxes[i])
        return result

    def get_state(self) -> dict:
        return {
            "fitted": self.fitted,
            "coefs": [r.coef_.tolist() for r in self.regs] if self.fitted else None,
            "intercepts": [float(r.intercept_) for r in self.regs] if self.fitted else None,
        }

    def set_state(self, state: dict) -> None:
        self.fitted = state["fitted"]
        if self.fitted and state["coefs"] is not None:
            dummy_X = np.zeros((1, len(state["coefs"][0])))
            dummy_y = np.zeros(1)
            for k, reg in enumerate(self.regs):
                reg.fit(dummy_X, dummy_y)
                reg.coef_ = np.array(state["coefs"][k], dtype=np.float64)
                reg.intercept_ = np.float64(state["intercepts"][k])
