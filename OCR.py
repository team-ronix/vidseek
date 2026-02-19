import easyocr
import cv2
import json

class OCR:
    def __init__(self, frames, video_path):
        self.frames = frames
        self.video_path = video_path
        self.inverted_index = {}
        self.reader = easyocr.Reader(['en'])
    
    def process_frames(self):
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
                # Preprocess frame for OCR
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # Run OCR
                print("       Running OCR...")
                result = self.reader.readtext(blur)
                print(f"       Detected {len(result)} text regions")
                
                for detection in result:
                    bbox, text, confidence = detection
                    print(f"       - '{text}' (confidence: {confidence:.2f})")
                    # Add to inverted index
                    if text not in self.inverted_index:
                        self.inverted_index[text] = []
                    
                    if text in self.inverted_index and any(occ['scene'] == scene.index for occ in self.inverted_index[text]):
                        continue  # Skip if already exists in this scene

                    self.inverted_index[text].append({
                        'scene': scene.index,
                        'frame': frame_number,
                        'video_path': self.video_path,
                        'start_time': scene.start_time,
                        'end_time': scene.end_time,
                        'confidence': confidence
                    })
            else:
                print("       Warning: Frame data is None")


    def save_inverted_index(self, output_path='inverted_index.json'):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.inverted_index, f, indent=2, ensure_ascii=False)
            
    def get_inverted_index(self):
        return self.inverted_index
        