from dataclasses import dataclass, field
import json
import os
import warnings
import numpy as np

@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_array(self):
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    @classmethod
    def from_array(cls, arr):
        return cls(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))

    @classmethod
    def from_yxyx(cls, y1, y2, x1, x2):
        return cls(float(x1), float(y1), float(x2), float(y2))

    def union(self, other):
        return BBox(
            min(self.x1, other.x1), min(self.y1, other.y1),
            max(self.x2, other.x2), max(self.y2, other.y2),
        )

    def iou(self, other):
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class DetectedObject:
    bbox: BBox
    label: str
    label_idx: int = -1
    score: float = 1.0


@dataclass
class RelationshipTriplet:
    subject: DetectedObject
    predicate: str
    predicate_idx: int
    object_: DetectedObject
    score: float = 1.0

    def __str__(self):
        return f"({self.subject.label}, {self.predicate}, {self.object_.label})"


@dataclass
class ImageAnnotation:
    image_id: str
    image_path: object  # can be None
    image_width: int
    image_height: int
    objects: list
    relationships: list


class VRDDataset:
    def __init__(
        self,
        annotation_path=None,
        object_classes=None,
        predicate_classes=None,
    ):
        _pred = predicate_classes if predicate_classes != None else []
        _obj = object_classes if object_classes != None else []
        self.predicate_classes = _pred
        self.object_classes = _obj
        self._pred2idx = {c: i for i, c in enumerate(_pred)}
        self._idx2pred = {i: c for c, i in self._pred2idx.items()}
        self._obj2idx = {c: i for i, c in enumerate(_obj)}
        self.annotations = {}
        if annotation_path != None:
            self._load(annotation_path)

    def _load(self, path):
        with open(path) as f:
            raw = json.load(f)
        unknown_preds = {}
        for img_id, ann in raw.items():
            objects = [
                DetectedObject(
                    bbox=BBox(*obj["bbox"]),
                    label=obj["label"],
                    label_idx=self._obj2idx.get(obj["label"], -1),
                    score=obj.get("score", 1.0),
                )
                for obj in ann["objects"]
            ]
            relationships = []
            for rel in ann.get("relationships", []):
                subj = objects[rel["subject_idx"]]
                obj_ = objects[rel["object_idx"]]
                pred = rel["predicate"]
                pred_idx = self._pred2idx.get(pred, -1)
                if pred_idx < 0:
                    # this predicate is not in our vocabulary
                    unknown_preds[pred] = unknown_preds.get(pred, 0) + 1
                relationships.append(RelationshipTriplet(
                    subject=subj,
                    predicate=pred,
                    predicate_idx=pred_idx,
                    object_=obj_,
                ))
            self.annotations[img_id] = ImageAnnotation(
                image_id=img_id,
                image_path=ann.get("path"),
                image_width=ann["width"],
                image_height=ann["height"],
                objects=objects,
                relationships=relationships,
            )
        if unknown_preds:
            total = sum(unknown_preds.values())
            names = sorted(unknown_preds, key=lambda k: -unknown_preds[k])
            print(f"[VRDDataset] Ignoring {total} relationships with unknown predicates ({len(names)} unique).")

    def add(self, annotation):
        self.annotations[annotation.image_id] = annotation

    def __len__(self):
        return len(self.annotations)

    def __iter__(self):
        return iter(self.annotations.values())

    def get_all_relationships(self):
        rels = []
        for ann in self.annotations.values():
            rels.extend(ann.relationships)
        return rels

    def predicate_distribution(self):
        dist = {}
        for rel in self.get_all_relationships():
            dist[rel.predicate] = dist.get(rel.predicate, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))
