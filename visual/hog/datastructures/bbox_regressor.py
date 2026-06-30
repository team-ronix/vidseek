import numpy as np
from sklearn.linear_model import Ridge


class BBoxRegressor:
    def __init__(self, alpha=1000.0):
        self.alpha = alpha
        self.regs = [Ridge(alpha=alpha) for _ in range(4)]
        self.fitted = False

    def fit(self, features, targets):
        if len(features) == 0:
            print("Warning: no features to fit bbox regressor")
            return
        for k in range(4):
            reg = self.regs[k]
            reg.fit(features, targets[:, k])
        self.fitted = True
        print(f"BBoxRegressor fitted with {len(features)} samples")

    def predict(self, feat):
        if not self.fitted:
            if feat.ndim == 1:
                return np.zeros(4, dtype=np.float32)
            else:
                return np.zeros((feat.shape[0], 4), dtype=np.float32)
        if feat.ndim == 1:
            feat = feat.reshape(1, -1)
        result = np.column_stack([reg.predict(feat) for reg in self.regs]).astype(np.float32)
        return result

    def get_state(self):
        state = {
            "fitted": self.fitted,
            "coefs": [r.coef_.tolist() for r in self.regs] if self.fitted else None,
            "intercepts": [float(r.intercept_) for r in self.regs] if self.fitted else None,
        }
        return state

    def set_state(self, state):
        self.fitted = state["fitted"]
        if self.fitted == True and state["coefs"] != None:
            X_dummy = np.zeros((1, len(state["coefs"][0])))
            y_dummy = np.zeros(1)
            for k in range(4):
                reg = self.regs[k]
                reg.fit(X_dummy, y_dummy)
                reg.coef_ = np.array(state["coefs"][k], dtype=np.float64)
                reg.intercept_ = np.float64(state["intercepts"][k])
