import numpy as np
from typing import List, Optional, Dict, Tuple
import time
from visual.vrd_ml.vrd_dataset import BBox, DetectedObject, RelationshipTriplet, VRDDataset, ImageAnnotation
from visual.vrd_ml.features.spatial import SpatialFeatureExtractor
from visual.vrd_ml.features.visual import VisualFeatureExtractor
from visual.vrd_ml.features.semantic import SemanticFeatureExtractor
from visual.vrd_ml.models.classifier import PredicateClassifier
from visual.vrd_ml.models.language_prior import LanguagePrior
from visual.vrd_ml.utils.pair_sampler import enumerate_pairs, sample_training_pairs


class VRDModel:
    def __init__(
        self,
        classifier_name: str = "svm",
        predicate_classes: Optional[List[str]] = None,
        glove_path: Optional[str] = None,
        min_score: float = 0.3,
        max_pairs: Optional[int] = None,
        top_k_predict: int = 5,
        use_language_prior: bool = True,
        prior_weight: float = 1.0,
        prior_topk_pool: int = 15,
        **clf_kwargs,
    ):
        self.classifier_name = classifier_name
        self.predicate_classes = predicate_classes if predicate_classes is not None else []
        self.glove_path = glove_path
        self.min_score = min_score
        self.max_pairs = max_pairs
        self.top_k_predict = top_k_predict
        self.prior_weight = prior_weight
        self.prior_topk_pool = prior_topk_pool

        self._pred2idx: Dict[str, int] = {
            p: i for i, p in enumerate(self.predicate_classes)
        }
        self._idx2pred: Dict[int, str] = {
            i: p for p, i in self._pred2idx.items()
        }

        self.spatial_ext = SpatialFeatureExtractor()
        self.visual_ext = VisualFeatureExtractor()
        self.semantic_ext = SemanticFeatureExtractor(glove_path=glove_path)

        self.feature_dim = (
            self.spatial_ext.dim +
            self.visual_ext.dim +
            self.semantic_ext.dim
        )

        self.clf = PredicateClassifier(
            classifier_name=classifier_name,
            label_names=self.predicate_classes,
            idx2pred=self._idx2pred,
            **clf_kwargs,
        )

        self.language_prior: Optional[LanguagePrior] = (
            LanguagePrior(predicate_classes=self.predicate_classes, glove_path=glove_path)
            if use_language_prior else None
        )

        self.seen_triplets: set = set()


    def _extract_pair_features(
        self,
        image: np.ndarray,
        subj: DetectedObject,
        obj: DetectedObject,
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        sp = self.spatial_ext.extract(subj.bbox, obj.bbox, img_w, img_h)
        vs = self.visual_ext.extract(image, subj.bbox, obj.bbox)
        sm = self.semantic_ext.extract(subj.label, obj.label)
        return np.concatenate([sp, vs, sm]).astype(np.float32)

    def _build_feature_matrix(
        self,
        dataset: VRDDataset,
        images: Dict[str, np.ndarray],
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_rows: List[np.ndarray] = []
        y_vals: List[int] = []
        rng = np.random.default_rng(42)

        n = len(dataset)
        t0 = time.time()

        for i, ann in enumerate(dataset):
            img = images.get(ann.image_id)
            if img is None:
                continue

            H, W = img.shape[:2]
            pairs, labels = sample_training_pairs(
                ann.objects,
                ann.relationships,
                neg_ratio=2.0,
                rng=rng,
            )

            for (subj, obj), label in zip(pairs, labels):
                feat = self._extract_pair_features(img, subj, obj, W, H)
                X_rows.append(feat)
                y_vals.append(label)
                if label >= 0:
                    self.seen_triplets.add(
                        (subj.label, self._idx2pred.get(label, "?"), obj.label)
                    )

            if verbose and (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{n}] features extracted  ({elapsed:.1f}s)", flush=True)

        if verbose:
            elapsed = time.time() - t0
            print(f"  [{n}/{n}] features extracted  ({elapsed:.1f}s)", flush=True)

        if not X_rows:
            raise ValueError("No training examples extracted. Check your dataset.")

        X = np.stack(X_rows).astype(np.float32)
        y = np.array(y_vals, dtype=np.int32)

        if verbose:
            pos = (y >= 0).sum()
            print(
                f"[model] Feature matrix: {X.shape}  "
                f"positives={pos:,}  negatives={len(y)-pos:,}"
            )

        return X, y


    def fit(
        self,
        dataset: VRDDataset,
        images: Dict[str, np.ndarray],
        verbose: bool = True,
    ) -> "VRDModel":
        if verbose:
            print(f"\n[model] Building feature matrix")
        X, y = self._build_feature_matrix(dataset, images, verbose)

        if verbose:
            print(f"\n[model] Training {self.classifier_name.upper()} classifier")
        self.clf.fit(X, y, verbose=verbose)

        if self.language_prior is not None:
            if verbose:
                print(f"\n[model] Fitting language/frequency prior")
            self.language_prior.fit(dataset)

        return self


    def predict(
        self,
        image: np.ndarray,
        boxes: List[List[float]],
        labels: List[str],
        scores: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        min_prob: float = 0.0,
    ) -> List[RelationshipTriplet]:
        if scores is None:
            scores = [1.0] * len(boxes)
        H, W = image.shape[:2]
        top_k = top_k or self.top_k_predict
        objects = [
            DetectedObject(bbox=BBox(*b), label=l, score=float(s))
            for b, l, s in zip(boxes, labels, scores)
        ]
        pairs = enumerate_pairs(
            objects,
            min_score=self.min_score,
            max_pairs=self.max_pairs,
        )
        if not pairs:
            return []
        feats = np.stack([
            self._extract_pair_features(image, subj, obj, W, H)
            for subj, obj in pairs
        ]).astype(np.float32)
        pool_k = max(top_k, self.prior_topk_pool) if self.language_prior is not None else top_k
        top_preds = self.clf.predict_topk(feats, k=pool_k)
        triplets: List[RelationshipTriplet] = []
        for (subj, obj), pair_preds in zip(pairs, top_preds):
            if self.language_prior is not None:
                rescored = []
                for pred_name, prob in pair_preds:
                    if prob < min_prob:
                        continue
                    prior_p = self.language_prior.score(subj.label, obj.label, pred_name)
                    combined = prob * (prior_p ** self.prior_weight)
                    rescored.append((pred_name, combined))
                rescored.sort(key=lambda x: -x[1])
                pair_preds = rescored[:top_k]

            for pred_name, prob in pair_preds:
                if pred_name not in self._pred2idx:
                    continue
                triplets.append(RelationshipTriplet(
                    subject=subj,
                    predicate=pred_name,
                    predicate_idx=self._pred2idx[pred_name],
                    object_=obj,
                    score=float(prob * subj.score * obj.score),
                ))

        triplets.sort(key=lambda t: -t.score)
        return triplets

    def predict_annotation(
        self,
        ann: ImageAnnotation,
        image: np.ndarray,
        min_prob: float = 0.0,
    ) -> List[Dict]:
        boxes = [[o.bbox.x1, o.bbox.y1, o.bbox.x2, o.bbox.y2] for o in ann.objects]
        labels = [o.label for o in ann.objects]
        scores = [o.score for o in ann.objects]
        triplets = self.predict(image, boxes, labels, scores, min_prob=min_prob)
        return [
            {
                "subj": t.subject.label,
                "pred": t.predicate,
                "obj": t.object_.label,
                "score": t.score,
                "subj_box": [t.subject.bbox.x1, t.subject.bbox.y1,
                             t.subject.bbox.x2, t.subject.bbox.y2],
                "obj_box":  [t.object_.bbox.x1, t.object_.bbox.y1,
                             t.object_.bbox.x2, t.object_.bbox.y2],
            }
            for t in triplets
        ]


    def save(self, path: str):
        import pickle, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "classifier_name": self.classifier_name,
            "predicate_classes": self.predicate_classes,
            "glove_path": self.glove_path,
            "min_score": self.min_score,
            "max_pairs": self.max_pairs,
            "top_k_predict": self.top_k_predict,
            "seen_triplets": self.seen_triplets,
            "use_language_prior": self.language_prior is not None,
            "prior_weight": self.prior_weight,
            "prior_topk_pool": self.prior_topk_pool,
            "language_prior_state": (
                self.language_prior.state_dict() if self.language_prior is not None else None
            ),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        self.clf.save(path.replace(".pkl", "_clf.pkl"))
        print(f"[model] Saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "VRDModel":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(
            classifier_name=state["classifier_name"],
            predicate_classes=state["predicate_classes"],
            glove_path=state.get("glove_path"),
            min_score=state["min_score"],
            max_pairs=state["max_pairs"],
            top_k_predict=state["top_k_predict"],
            use_language_prior=state.get("use_language_prior", False),
            prior_weight=state.get("prior_weight", 1.0),
            prior_topk_pool=state.get("prior_topk_pool", 15),
        )
        obj.seen_triplets = state["seen_triplets"]
        obj.clf = PredicateClassifier.load(path.replace(".pkl", "_clf.pkl"))
        if obj.language_prior is not None and state.get("language_prior_state") is not None:
            obj.language_prior = LanguagePrior.from_state_dict(
                state["language_prior_state"], glove_path=state.get("glove_path")
            )
        print(f"[model] Loaded ← {path}")
        return obj
