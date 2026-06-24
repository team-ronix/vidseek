import os
import json
import torch
from visual.faster_rcnn.model.faster_rcnn import FasterRCNN
from visual.faster_rcnn.voc_dataset import VOC_CLASSES
from PIL import Image
import numpy as np

_ObjDetector_DIR = os.path.dirname(os.path.abspath(__file__))
_FASTER_RCNN_MODEL_PATH = os.path.join(_ObjDetector_DIR, 'faster_rcnn_final.pth')

class ObjectDetector:
    def __init__(self, video_path, frames, score_thresh=0.6):
        self.video_path = video_path
        self.frames = frames
        self.inverted_index = {}
        NUM_CLASSES = len(VOC_CLASSES)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FasterRCNN(num_classes=NUM_CLASSES, score_thresh=score_thresh, pretrained=False).to(self.device)
        if not os.path.exists(_FASTER_RCNN_MODEL_PATH):
            raise ValueError("Model file does not exist")
        self.model.load_state_dict(torch.load(_FASTER_RCNN_MODEL_PATH, map_location=self.device))
        self.model.eval()
        
    def _process_img(self, img, target_size=600, max_size=1000):
        pixel_mean = np.array([0.485, 0.456, 0.406], np.float32)
        pixel_std = np.array([0.229, 0.224, 0.225], np.float32)
        w, h = img.size
        scale = target_size / min(h, w)
        if scale * max(h, w) > max_size:
            scale = max_size / max(h, w)
        img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
        arr = (np.array(img, np.float32) / 255.0 - pixel_mean) / pixel_std
        return torch.from_numpy(arr.copy()).permute(2, 0, 1), scale
    
    def _build_results(self, frame):
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        processed_frame, scale = self._process_img(frame)
        scaled_H, scaled_W = processed_frame.shape[1], processed_frame.shape[2]
        with torch.no_grad():
            res = self.model(processed_frame.unsqueeze(0).to(self.device), [(scaled_H, scaled_W)])[0]
        pred_boxes  = res['boxes'].cpu()
        pred_scores = res['scores'].cpu()
        pred_labels = res['labels'].cpu()
        print(f'\nDetections ({len(pred_scores)} found):')
        results = []
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            cls_name = VOC_CLASSES[label.item() - 1]
            x1, y1, x2, y2 = (v / scale for v in box.tolist())
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