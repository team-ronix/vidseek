import os
import pickle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def _build_classifier(name, **kwargs):
    name = name.lower()
    if name == "svm":
        base = LinearSVC(
            C=kwargs.get("C", 1.0),
            class_weight="balanced",
            random_state=42,
            max_iter=kwargs.get("max_iter", 2000),
        )
        return CalibratedClassifierCV(base, cv=3)
    elif name == "rf":
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 50),
            max_depth=kwargs.get("max_depth", None),
            max_features=kwargs.get("max_features", "sqrt"),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif name == "lr":
        return LogisticRegression(
            C=kwargs.get("C", 1.0),
            solver="saga",
            penalty="l2",
            max_iter=kwargs.get("max_iter", 1000),
            tol=kwargs.get("tol", 1e-3),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown classifier '{name}")


class PredicateClassifier:
    def __init__(
        self,
        classifier_name="svm",
        label_names=None,
        idx2pred=None,
        **clf_kwargs,
    ):
        self.classifier_name = classifier_name
        self.label_names = label_names if label_names != None else []
        self.n_classes = len(self.label_names)
        self.label2idx = {l: i for i, l in enumerate(self.label_names)}
        if idx2pred != None:
            self._idx2pred = idx2pred
        else:
            self._idx2pred = {i: l for i, l in enumerate(self.label_names)}

        self._clf_kwargs = clf_kwargs
        self.pipeline = None
        self.is_fitted = False
        self.feature_dim = None


    def fit(self, X, y, verbose=True):
        self.feature_dim = X.shape[1]
        mask = y >= 0
        X_train = X[mask]
        y_train = y[mask]
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            raise ValueError(
                "Need at least 2 predicate classes in training data. "
                "Check that your dataset has diverse relationships."
            )
        if verbose == True:
            print(f"[classifier] Training {self.classifier_name.upper()} on {X_train.shape[0]} samples, "
                  f"{len(unique_classes)} classes, dim={X_train.shape[1]}")
        clf = _build_classifier(self.classifier_name, **self._clf_kwargs)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        if verbose == True:
            y_pred = self.pipeline.predict(X_train)
            train_acc = (y_pred == y_train).mean()
            print(f"[classifier] Training accuracy: {train_acc:.3f}")
        return self

    def predict(self, X):
        self._check_fitted()
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        self._check_fitted()
        return self.pipeline.predict_proba(X)


    def predict_topk(self, X, k=5):
        self._check_fitted()
        proba = self.predict_proba(X)  # shape: (N, C)
        classes = self.pipeline.classes_  # int indices seen during fit
        result = []
        for row in proba:
            top_local = np.argsort(row)[::-1][:k]
            top = [
                (
                    self._idx2pred.get(int(classes[li]), f"pred_{classes[li]}"),
                    float(row[li]),
                )
                for li in top_local
            ]
            result.append(top)
        return result

    def evaluate(self, X, y):
        self._check_fitted()
        mask = y >= 0
        X_eval = X[mask]
        y_eval = y[mask]
        y_pred = self.pipeline.predict(X_eval)
        acc = float((y_pred == y_eval).mean())
        classes_seen = self.pipeline.classes_
        names = [self._idx2pred.get(int(c), str(c)) for c in classes_seen]
        report = classification_report(
            y_eval, y_pred,
            labels=classes_seen,
            target_names=names,
            zero_division=0,
            output_dict=True,
        )
        return {"accuracy": acc, "classification_report": report}

    def save(self, path):
        self._check_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "pipeline": self.pipeline,
                "classifier_name": self.classifier_name,
                "label_names": self.label_names,
                "idx2pred": self._idx2pred,
                "feature_dim": self.feature_dim,
            }, f)
        print(f"[classifier] Saved to {path}")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(
            classifier_name=state["classifier_name"],
            label_names=state["label_names"],
            idx2pred=state.get("idx2pred"),
        )
        obj.pipeline = state["pipeline"]
        obj.feature_dim = state["feature_dim"]
        obj.is_fitted = True
        print(f"[classifier] Loaded from {path}")
        return obj

    def _check_fitted(self):
        if self.is_fitted != True:
            raise RuntimeError("Classifier not fitted yet. Call .fit() first.")
