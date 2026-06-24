import os
import json
from visual.hog.hog_detector import HOGDetector
from PIL import Image
import numpy as np

_ObjDetector_DIR = os.path.dirname(os.path.abspath(__file__))
_HOG_MODEL_PATH = os.path.join(_ObjDetector_DIR, 'model')

class ObjectDetector:
    def __init__(self, video_path, frames, score_thresh=0.8, top_k=5, use_context=False):
        self.video_path = video_path
        self.frames = frames
        self.inverted_index = {}
        self.score_thresh = score_thresh
        self.top_k = top_k
        self.use_context = use_context
        self.detector = HOGDetector(
            classes=[],
            hog_descriptor_params=dict(
                cell_size=8, n_orient_cs=18, n_orient_ci=9,
                alpha=0.2, n_energy=4, n_octaves=4, llambda=4, min_size=48,
            ),
        )
        if not os.path.exists(_HOG_MODEL_PATH):
            raise ValueError("Model file does not exist")
        self.detector.load(_HOG_MODEL_PATH)
        
    
    def _build_results(self, frame):
        pred_boxes, pred_scores, pred_labels = self.detector.detect(
            frame,
            threshold = self.score_thresh,
            overlap_threshold = 0.5,
            use_context = self.use_context,
        )
        print(f'\nDetections ({len(pred_scores)} found):')
        results = []
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            cls_name = label
            x1, y1, x2, y2 = box
            results.append((cls_name, score, (x1, y1, x2, y2)))
        return results

    def detect_objects(self):
        for i, frame_data in enumerate(self.frames, 1):
            frame_number = frame_data['frame_number']
            frame_time = frame_data['frame_time']
            scene = frame_data['scene']
            frame_count = frame_data['frame_count_in_scene']
            frame = frame_data['frame']
            
            print(f"[{i}/{len(self.frames)}] Scene {scene.index}: Start at {scene.start_time:.2f}s, End at {scene.end_time:.2f}s, Duration {scene.duration:.2f}s ({frame_count} frames)")
            print(f"\tProcessing frame {frame_number}")
            
            if frame_number is None:
                continue
            
            if frame is not None:
                results = self._build_results(frame)
                results.sort(key=lambda x: x[1], reverse=True)  # Sort results by score
                results = results[:self.top_k]  # Keep only top_k results
                for name, conf, box in results:
                        print(f"       - Detected '{name}' with confidence {conf:.2f}")
                        if name not in self.inverted_index:
                            self.inverted_index[name] = []
                        
                        # Skip if already exists in this scene
                        if any(occ['scene'] == scene.index for occ in self.inverted_index[name]):
                            continue

                        self.inverted_index[name].append({
                            'scene': scene.index,
                            'frame': frame_number,
                            'video_path': self.video_path,
                            "frame_time": frame_time,
                            'start_time': scene.start_time,
                            'end_time': scene.end_time,
                            'confidence': float(conf)
                        })
            

    def save_inverted_index(self, output_path='object_inverted_index.json'):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.inverted_index, f, indent=2, ensure_ascii=False)
            
    def load_inverted_index(self, input_path='object_inverted_index.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            self.inverted_index = json.load(f)
            
    def get_inverted_index(self):
        return self.inverted_index