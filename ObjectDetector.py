from ultralytics import YOLO
import json

class ObjectDetector:
    def __init__(self, video_path, frames, model_name="yolo26s.pt"):
        self.video_path = video_path
        self.frames = frames
        self.model = YOLO(model_name)
        self.inverted_index = {}

    def detect_objects(self):
        for i, frame_data in enumerate(self.frames, 1):
            frame_number = frame_data['frame_number']
            scene = frame_data['scene']
            frame_count = frame_data['frame_count_in_scene']
            frame = frame_data['frame']
            
            print(f"[{i}/{len(self.frames)}] Scene {scene.index}: Start at {scene.start_time:.2f}s, End at {scene.end_time:.2f}s, Duration {scene.duration:.2f}s ({frame_count} frames)")
            print(f"       Processing frame {frame_number}")
            
            if frame_number is None:
                continue
            
            if frame is not None:    
                results = self.model(frame)
                results.sort(key=lambda x: x.boxes.conf, reverse=True)  # Sort results by confidence
                for result in results:
                    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
                    confs = result.boxes.conf  # confidence score of each box
                    for (name, conf) in zip(names, confs):
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
                            'start_time': scene.start_time,
                            'end_time': scene.end_time,
                            'confidence': float(conf)
                        })
            

    def save_inverted_index(self, output_path='object_inverted_index.json'):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.inverted_index, f, indent=2, ensure_ascii=False)
            
    def get_inverted_index(self):
        return self.inverted_index