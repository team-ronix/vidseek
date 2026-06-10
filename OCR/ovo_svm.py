import numpy as np
import os
import joblib
from itertools import combinations
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed


class BinarySVM:
    def __init__(self, lr=0.001, C=1.0, epochs=500):
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.w = None
        self.b = 0

    def fit(self, X, y):
        # y in {-1, +1}
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            indices = np.random.permutation(n_samples)

            for i in indices:
                xi = X[i]
                yi = y[i]

                margin = yi * (np.dot(xi, self.w) + self.b)

                # L2 regularization gradient (standard form)
                if margin >= 1:
                    self.w -= self.lr * (self.w)
                else:
                    self.w -= self.lr * (self.w - self.C * yi * xi)
                    self.b += self.lr * self.C * yi

    def decision_function(self, X):
        return X @ self.w + self.b


class PlattScaler:
    def __init__(self):
        self.A = 0.0
        self.B = 0.0

    def _sigmoid(self, z):
        z = np.clip(z, -50, 50)
        return 1 / (1 + np.exp(-z))

    def fit(self, scores, labels, lr=0.001, epochs=500):
        scores = np.array(scores)
        labels = np.array(labels)

        for _ in range(epochs):
            for f, y in zip(scores, labels):
                z = self.A * f + self.B
                p = self._sigmoid(z)

                error = p - y

                self.A -= lr * error * f
                self.B -= lr * error

    def predict_proba(self, scores):
        scores = np.array(scores)
        return self._sigmoid(self.A * scores + self.B)


class OvO_SVM:
    def __init__(self, lr=0.001, C=1.0, epochs=500):
        self.lr = lr
        self.C = C
        self.epochs = epochs

        self.models = {}
        self.platt = {}
        self.classes = None

    @staticmethod
    def train_pair(i, j, X, y, lr, C, epochs, save_path):

        svm_path = os.path.join(save_path, f"svm_{i}_{j}.joblib")
        platt_path = os.path.join(save_path, f"platt_{i}_{j}.joblib")

        if os.path.exists(svm_path) and os.path.exists(platt_path):
            print(f"[SKIP] {i} vs {j}")
            return (
                (i, j),
                joblib.load(svm_path),
                joblib.load(platt_path)
            )

        print(f"[TRAIN] {i} vs {j}")

        mask = (y == i) | (y == j)
        X_ij = X[mask]
        y_ij = y[mask]

        y_bin = np.where(y_ij == i, 1, -1)

        X_tr, X_cal, y_tr, y_cal, y_bin_tr, y_bin_cal = train_test_split(
            X_ij,
            y_ij,
            y_bin,
            test_size=0.3,
            random_state=42,
            shuffle=True,
        )

        svm = BinarySVM(lr, C, epochs)
        svm.fit(X_tr, y_bin_tr)

        cal_scores = svm.decision_function(X_cal)
        cal_labels = (y_cal == i).astype(int)

        platt = PlattScaler()
        platt.fit(cal_scores, cal_labels)

        joblib.dump(svm, svm_path)
        joblib.dump(platt, platt_path)

        print(f"[DONE] {i} vs {j}")

        return ((i, j), svm, platt)

    def fit(self, X, y, save_path):
        self.classes = np.unique(y)
        pairs = list(combinations(self.classes, 2))

        os.makedirs(save_path, exist_ok=True)

        meta = {
            "lr": self.lr,
            "C": self.C,
            "epochs": self.epochs,
            "pairs": pairs,
            "classes": list(self.classes),
        }

        joblib.dump(meta, os.path.join(save_path, "meta.joblib"))

        results = Parallel(
            n_jobs=-1,
            backend="loky",
            verbose=10,
        )(
            delayed(OvO_SVM.train_pair)(
                i, j, X, y,
                self.lr,
                self.C,
                self.epochs,
                save_path,
            )
            for i, j in pairs
        )

        for (i, j), svm, platt in results:
            self.models[(i, j)] = svm
            self.platt[(i, j)] = platt

    def predict_proba(self, X):
        class_scores = {c: np.zeros(len(X)) for c in self.classes}

        for i, j in combinations(self.classes, 2):
            svm = self.models[(i, j)]
            platt = self.platt[(i, j)]

            scores = svm.decision_function(X)
            p_i = platt.predict_proba(scores)

            class_scores[i] += p_i
            class_scores[j] += (1 - p_i)

        probs = np.vstack(list(class_scores.values())).T
        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.asarray(self.classes)[np.argmax(probs, axis=1)]

    def confidence(self, X):
        return np.max(self.predict_proba(X), axis=1)

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        meta = {
            "classes": self.classes,
            "lr": self.lr,
            "C": self.C,
            "epochs": self.epochs,
            "pairs": list(self.models.keys()),
        }

        joblib.dump(meta, os.path.join(path, "meta.joblib"))

        for (i, j), model in self.models.items():
            joblib.dump(model, os.path.join(path, f"svm_{i}_{j}.joblib"))

        for (i, j), scaler in self.platt.items():
            joblib.dump(scaler, os.path.join(path, f"platt_{i}_{j}.joblib"))

    @classmethod
    def load(cls, path):
        meta = joblib.load(os.path.join(path, "meta.joblib"))

        obj = cls(
            lr=meta["lr"],
            C=meta["C"],
            epochs=meta["epochs"],
        )

        obj.classes = np.asarray(meta["classes"])
        obj.models = {}
        obj.platt = {}

        for (i, j) in meta["pairs"]:
            obj.models[(i, j)] = joblib.load(os.path.join(path, f"svm_{i}_{j}.joblib"))
            obj.platt[(i, j)] = joblib.load(os.path.join(path, f"platt_{i}_{j}.joblib"))

        return obj
