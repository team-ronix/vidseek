import os
import json
import time
from PIL import Image
import cv2
from visual.vrd_ml.models.vrd_model import VRDModel

_VRD_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_VRD_DIR, 'checkpoints', 'vrd_rf_real.pkl')


class VRD:
    def __init__(self, video_path, frames, objects, top_k=5):
        self.video_path = video_path
        self.frames = frames
        self.objects = objects
        self.top_k = top_k
        if not os.path.exists(_MODEL_PATH):
            raise ValueError("Model file does not exist")
        self.model = VRDModel.load(_MODEL_PATH)
        self.inverted_index = {}

    def _build_pred_data(self, frame_data, obj_data):
        boxes = []
        labels = []
        scores = []
        for obj in obj_data:
            label, score, bbox = obj
            x1, y1, x2, y2 = bbox
            boxes.append([x1, y1, x2, y2])
            labels.append(label)
            scores.append(score)
        return self.model.predict(frame_data, boxes, labels, scores, top_k=1)

    def detect_relationships(self):
        for i, (frame_data, obj_data) in enumerate(zip(self.frames, self.objects), 1):
            frame_number = frame_data['frame_number']
            frame_time = frame_data['frame_time']
            scene = frame_data['scene']
            frame_count = frame_data['frame_count_in_scene']
            frame = frame_data['frame']

            print(f"[{i}/{len(self.frames)}] Scene {scene.index}: Start at {scene.start_time:.2f}s, End at {scene.end_time:.2f}s, Duration {scene.duration:.2f}s ({frame_count} frames)")
            print(f"       Processing frame {frame_number}")

            if frame_number is None:
                continue

            triples = self._build_pred_data(frame, obj_data)
            print(f"       Found {len(triples)} relationships in frame {frame_number}")
            for triple in triples:
                subject = triple.subject.label
                predicate = triple.predicate
                object_ = triple.object_.label
                print(f"       - Detected relationship: ({subject}, {predicate}, {object_})")
                statement = f"{subject}, {predicate}, {object_}"
                if statement not in self.inverted_index:
                    self.inverted_index[statement] = []

                if any(occ['scene'] == scene.index for occ in self.inverted_index[statement]):
                    continue

                self.inverted_index[statement].append({
                    'scene': scene.index,
                    'frame': frame_number,
                    'video_path': self.video_path,
                    'frame_time': frame_time,
                    'start_time': scene.start_time,
                    'end_time': scene.end_time
                })

    def save_inverted_index(self, output_path='vrd_inverted_index.json'):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.inverted_index, f, indent=2, ensure_ascii=False)

    def load_inverted_index(self, input_path='vrd_inverted_index.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            self.inverted_index = json.load(f)

    def get_inverted_index(self):
        return self.inverted_index
