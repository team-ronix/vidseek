import numpy as np
from sklearn.linear_model import Ridge


class BBoxRegressor:
    def __init__(self, alpha: float = 1000.0):
        self.alpha = alpha
        self.regs = [Ridge(alpha=alpha) for _ in range(4)]
        self.fitted = False

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
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
        return np.column_stack([reg.predict(feat) for reg in self.regs]).astype(np.float32)

    def get_state(self) -> dict:
        return {
            "fitted": self.fitted,
            "coefs": [r.coef_.tolist() for r in self.regs] if self.fitted else None,
            "intercepts": [float(r.intercept_) for r in self.regs] if self.fitted else None,
        }

    def set_state(self, state: dict) -> None:
        self.fitted = state["fitted"]
        if self.fitted and state["coefs"] is not None:
            X = np.zeros((1, len(state["coefs"][0])))
            y = np.zeros(1)
            for k, reg in enumerate(self.regs):
                reg.fit(X, y)
                reg.coef_ = np.array(state["coefs"][k], dtype=np.float64)
                reg.intercept_ = np.float64(state["intercepts"][k])
